"""Modal side of the pipeline.

A job is a YAML file in ops/jobs/ listing steps; a list inside `steps` runs in parallel.
GitHub Actions runs `modal run modal_app.py --spec ops/jobs/<id>.yaml` and then copies
/vol/results/<id>/ into the repo at ops/results/<id>/.

Only model outputs, metrics and logs go to /vol/results. Human-written text lives in
/vol/corpus.jsonl and /vol/data/ and is never exported.

    /vol/models/base        Qwen/Qwen3.8-27B
    /vol/models/<name>      merged checkpoints from train steps
    /vol/adapters/<name>    LoRA adapters
    /vol/corpus.jsonl       scraped essays (not exported)
    /vol/data/*.jsonl       training sets (not exported)
    /vol/results/<job>/     exported to the repo

Paths in step args: "vol:x" is /vol/x, "results:<job>/f" is another job's result, anything
else is a repo path.
"""
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import modal

VOL = "/vol"
REPO = "/repo"
vol = modal.Volume.from_name("owl-vol", create_if_missing=True)
hf_cache = modal.Volume.from_name("owl-hf-cache", create_if_missing=True)
MOUNTS = {VOL: vol, "/hf": hf_cache}
GPU = os.environ.get("OWL_GPU", "H200")


def with_repo(img):
    return (img.env({"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_python_source("owl")
        .add_local_dir("configs", f"{REPO}/configs")
        .add_local_dir("data", f"{REPO}/data"))


cpu_image = with_repo(modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "httpx==0.28.1", "trafilatura==2.0.0", "beautifulsoup4==4.13.4", "lxml==5.4.0",
    "pyyaml==6.0.2", "huggingface_hub>=0.34", "hf_transfer"))
train_image = with_repo(
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("torch==2.14.0", "torchvision==0.29.0",
        index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("transformers==5.22.2", "trl==1.17.1", "peft==0.21.2",
        "accelerate==1.15.0", "datasets", "flash-linear-attention==0.6.0", "safetensors"))
vllm_image = with_repo(modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "vllm==0.33.0", "pyyaml"))
app = modal.App("onwritingwell")


def path(p):
    if p.startswith("vol:"):
        return f"{VOL}/{p[4:]}"
    if p.startswith("results:"):
        return f"{VOL}/results/{p[8:]}"
    return f"{REPO}/{p}"


def results(job):
    d = Path(VOL, "results", job)
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(cmd, log):
    print("+", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=REPO, env={**os.environ, "PYTHONPATH": "/root"})
    with open(log, "a") as f:
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
    assert p.wait() == 0, f"command failed: {' '.join(cmd)}"


@app.function(image=cpu_image, volumes=MOUNTS, timeout=3 * 3600)
def download_base(job, args):
    from huggingface_hub import snapshot_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if Path(VOL, "models/base/config.json").exists() and not args.get("force"):
        return {"status": "exists"}
    snapshot_download(args.get("repo", "Qwen/Qwen3.8-27B"), local_dir=f"{VOL}/models/base")
    vol.commit()
    return {"status": "downloaded", "files": sorted(os.listdir(f"{VOL}/models/base"))}


@app.function(image=cpu_image, volumes=MOUNTS, timeout=6 * 3600)
def build_corpus(job, args):
    """Fetch every manifest URL; flag docs whose text hash differs from the one annotated,
    since their paragraph indices may have shifted."""
    import concurrent.futures as cf

    from owl.scrape import fetch_doc

    manifest = [json.loads(l) for l in open(path(args.get("manifest", "data/manifest.jsonl")))]
    stats = Counter()
    with open(f"{VOL}/corpus.jsonl", "w") as f, cf.ThreadPoolExecutor(8) as ex:
        docs = ex.map(lambda m: fetch_doc(m["source"], m["url"]), manifest)
        for m, doc in zip(manifest, docs):
            if doc is None:
                stats["missing"] += 1
                continue
            doc["hash_mismatch"] = doc["sha256"] != m["sha256"]
            stats["hash_mismatch" if doc["hash_mismatch"] else "ok"] += 1
            stats["source:" + doc["source"]] += 1
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    vol.commit()
    stats["total"] = len(manifest)
    (results(job) / "corpus_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


@app.function(image=cpu_image, volumes=MOUNTS, timeout=3600)
def build_sft(job, args):
    """Join annotator units with the corpus into chat examples, weight sources per
    configs/sources.yaml, optionally mix in retention examples. Held-out docs and docs whose
    text changed since annotation are dropped."""
    import yaml

    from owl.units import load_units, span_text, validate_unit

    corpus = {d["doc_id"]: d for d in map(json.loads, open(f"{VOL}/corpus.jsonl"))}
    holdout = set(json.load(open(path(args["holdout"])))) if args.get("holdout") else set()
    examples, errors = [], []
    for u in load_units(path(args.get("units", "data/backtranslations"))):
        doc = corpus.get(u["doc_id"])
        if u.get("skip") or u["doc_id"] in holdout or doc and doc["hash_mismatch"]:
            continue
        err = validate_unit(u, doc)
        text = "" if err else span_text(doc, u["para_start"], u["para_end"])
        err = err or (len(text.split()) > args.get("max_words", 4500) and "span too long")
        if err:
            errors.append(f"{u['doc_id']}:{u.get('para_start')}: {err}")
            continue
        examples.append({"id": f"{u['doc_id']}:{u['para_start']}-{u['para_end']}",
            "doc_id": u["doc_id"], "domain": u["domain"], "kind": u["kind"],
            "source": doc["source"], "messages": [
                {"role": "user", "content": u["prompt"].strip()},
                {"role": "assistant", "content": text}]})
    # Integer part of a source weight repeats the example, the fraction samples once more.
    weights = {s["name"]: s.get("weight", 1.0)
        for s in yaml.safe_load(open(f"{REPO}/configs/sources.yaml"))["sources"]}
    g = random.Random(0)
    mixed = []
    for e in examples:
        w = weights.get(e["source"], 1.0)
        mixed += [e] * (int(w) + (g.random() < w - int(w)))
    if args.get("retention"):
        ret = [json.loads(l) for l in open(path(args["retention"]))]
        g.shuffle(ret)
        mixed += [{"id": r["id"], "domain": "retention", "source": "retention", "messages": [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["response"]}]}
            for r in ret[:int(len(mixed) * args.get("retention_frac", 0.25))]]
    g.shuffle(mixed)
    out = Path(VOL, "data", args.get("out", "sft.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in mixed))
    vol.commit()
    stats = {"examples": len(examples), "after_weighting": len(mixed),
        "n_errors": len(errors), "errors": errors[:200],
        "by_domain": Counter(e["domain"] for e in mixed),
        "by_source": Counter(e["source"] for e in mixed),
        "mean_words": sum(len(e["messages"][-1]["content"].split()) for e in mixed)
            / max(len(mixed), 1)}
    (results(job) / f"sft_stats_{out.stem}.json").write_text(json.dumps(stats, indent=1))
    # Prompts only, never the human text, so they can be reviewed in the repo.
    (results(job) / f"sft_prompts_{out.stem}.jsonl").write_text("".join(
        json.dumps({"id": e["id"], "domain": e["domain"], "prompt": e["messages"][0]["content"]},
            ensure_ascii=False) + "\n" for e in examples))
    vol.commit()
    return {k: v for k, v in stats.items() if k != "errors"}


@app.function(image=cpu_image, volumes=MOUNTS, timeout=3600)
def build_human_pairs(job, args):
    """Offline DPO pairs: chosen = human text, rejected = the model's answer to the same
    request. Stays on the volume because it contains human text."""
    sft = {e["id"]: e for e in map(json.loads, open(path(args["sft"])))
        if e["domain"] != "retention"}
    pairs = [{"id": s["id"], "prompt": sft[s["id"]]["messages"][0]["content"],
        "chosen": sft[s["id"]]["messages"][-1]["content"], "rejected": s["response"]}
        for s in map(json.loads, open(path(args["samples"])))
        if s["id"] in sft and s["finish_reason"] == "stop"]
    Path(VOL, "data", args.get("out", "human_pairs.jsonl")).write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs))
    vol.commit()
    return {"pairs": len(pairs)}


@app.function(image=vllm_image, gpu=GPU, volumes=MOUNTS, timeout=6 * 3600)
def generate(job, args):
    from owl.slop import aggregate

    vol.reload()
    rd = results(job)
    out = rd / args["out"]
    cmd = [sys.executable, "-m", "owl.generate", "--model", f"{VOL}/models/{args['model']}",
        "--prompts", path(args["prompts"]), "--out", str(out),
        "--tag", args.get("tag", args["model"]), "--n", str(args.get("n", 1)),
        "--temperature", str(args.get("temperature", 0.8)),
        "--max-tokens", str(args.get("max_tokens", 4096)), "--tp", str(args.get("tp", 1))]
    run(cmd + (["--limit", str(args["limit"])] if args.get("limit") else []),
        rd / "generate.log")
    rows = [json.loads(l) for l in open(out)]
    if args.get("private"):
        # e.g. answers to prompts for held-out human text: keep them off the repo
        out = out.rename(Path(VOL, "data", args["out"]))
    (rd / f"{Path(args['out']).stem}_summary.json").write_text(json.dumps({"rows": len(rows),
        "truncated": sum(r["finish_reason"] != "stop" for r in rows),
        "slop": aggregate([r["response"] for r in rows])}, indent=1))
    vol.commit()
    return {"rows": len(rows), "out": str(out)}


@app.function(image=train_image, gpu=GPU, volumes=MOUNTS, timeout=24 * 3600,
    memory=200 * 1024)
def train(job, args):
    """args: {stage: sft|dpo, model, data, out_model, hp: {...}}. Trains LoRA, then merges."""
    vol.reload()
    rd = results(job)
    name = args["out_model"]
    model = f"{VOL}/models/{args.get('model', 'base')}"
    cmd = [sys.executable, "-m", "owl.train", args["stage"], "--model", model,
        "--data", path(args["data"]), "--out", f"{VOL}/adapters/{name}"]
    for k, v in (args.get("hp") or {}).items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    run(cmd, rd / f"train_{name}.log")
    vol.commit()
    run([sys.executable, "-m", "owl.train", "merge", "--model", model,
        "--adapter", f"{VOL}/adapters/{name}", "--out", f"{VOL}/models/{name}"],
        rd / f"merge_{name}.log")
    shutil.copy(f"{VOL}/adapters/{name}/log_history.json", rd / f"log_history_{name}.json")
    vol.commit()
    return {"model": name}


@app.function(image=cpu_image, volumes=MOUNTS, timeout=600)
def ls(job, args):
    out = {sub: sorted(os.listdir(f"{VOL}/{sub}")) if Path(VOL, sub).exists() else []
        for sub in ("models", "data", "adapters", "results")}
    (results(job) / "ls.json").write_text(json.dumps(out, indent=1))
    vol.commit()
    return out


STEPS = {"download_base": download_base, "build_corpus": build_corpus,
    "build_sft": build_sft, "build_human_pairs": build_human_pairs, "generate": generate,
    "train": train, "ls": ls}


@app.local_entrypoint()
def main(spec):
    import yaml

    job = yaml.safe_load(open(spec))
    report = {"id": job["id"], "steps": []}
    for group in job["steps"]:
        calls = []
        for (name, args), in (s.items() for s in (group if isinstance(group, list) else [group])):
            print(f"[job {job['id']}] start {name} {json.dumps(args or {})}", flush=True)
            calls.append((name, STEPS[name].spawn(job["id"], args or {})))
        for name, call in calls:
            try:
                res = call.get()
            except Exception as e:
                report["steps"].append({"step": name, "ok": False, "error": repr(e)[:4000]})
                Path("job_report.json").write_text(json.dumps(report, indent=1, default=str))
                raise
            report["steps"].append({"step": name, "ok": True, "result": res})
            print(f"[job {job['id']}] done {name}: {json.dumps(res, default=str)[:2000]}",
                flush=True)
    Path("job_report.json").write_text(json.dumps(report, indent=1, default=str))
