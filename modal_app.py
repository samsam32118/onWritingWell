"""Modal side of the pipeline.

A *job* is a YAML file in ops/jobs/ with a list of steps. GitHub Actions runs

    modal run modal_app.py --spec ops/jobs/<id>.yaml

and afterwards copies /vol/results/<id>/ back into the repo at ops/results/<id>/.

Only model outputs, metrics and logs are ever written to /vol/results. Human-written corpus
text lives in /vol/corpus.jsonl and /vol/data/ and is never exported.

Volume layout:
    /vol/models/base             Qwen/Qwen3.8-27B snapshot
    /vol/models/<name>           merged checkpoints produced by train steps
    /vol/corpus.jsonl            scraped essays (not exported)
    /vol/data/*.jsonl            training sets (not exported)
    /vol/results/<job id>/       exported to the repo
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

APP = "onwritingwell"
BASE_MODEL = "Qwen/Qwen3.8-27B"
VOL = "/vol"
REPO = "/repo"

vol = modal.Volume.from_name("owl-vol", create_if_missing=True)
hf_cache = modal.Volume.from_name("owl-hf-cache", create_if_missing=True)
SECRETS = [modal.Secret.from_name("huggingface", required_keys=[])] if os.environ.get("OWL_HF_SECRET") else []


def _with_repo(img: modal.Image) -> modal.Image:
    return (
        img.env({"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_python_source("owl")
        .add_local_dir("configs", f"{REPO}/configs")
        .add_local_dir("data", f"{REPO}/data")
    )


cpu_image = _with_repo(
    modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "httpx==0.28.1", "trafilatura==2.0.0", "beautifulsoup4==4.13.4", "lxml==5.4.0", "pyyaml==6.0.2", "huggingface_hub>=0.34", "hf_transfer"
    )
)

train_image = _with_repo(
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .uv_pip_install(
        "transformers==5.22.2",
        "trl==1.17.1",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "datasets",
        "flash-linear-attention==0.6.0",
        "huggingface_hub>=0.34",
        "hf_transfer",
        "pyyaml",
        "safetensors",
    )
)

vllm_image = _with_repo(
    modal.Image.debian_slim(python_version="3.12").uv_pip_install("vllm==0.33.0", "pyyaml", "hf_transfer")
)

app = modal.App(APP)
MOUNTS = {VOL: vol, "/hf": hf_cache}


def _model_path(name: str) -> str:
    return f"{VOL}/models/{name}"


def _resolve(path: str) -> str:
    """'vol:x' -> /vol/x ; 'results:<job>/f' -> /vol/results/<job>/f ; otherwise a repo path."""
    if path.startswith("vol:"):
        return f"{VOL}/{path[4:]}"
    if path.startswith("results:"):
        return f"{VOL}/results/{path[8:]}"
    return f"{REPO}/{path}"


def _results_dir(job_id: str) -> Path:
    d = Path(VOL, "results", job_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run(cmd: list[str], log: Path):
    print("+", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": "/root"})
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        if p.wait() != 0:
            raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")


# ---------------------------------------------------------------- steps


@app.function(image=cpu_image, volumes=MOUNTS, timeout=3 * 3600, secrets=SECRETS)
def download_base(job_id: str, args: dict):
    from huggingface_hub import snapshot_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    dst = _model_path("base")
    if Path(dst, "config.json").exists() and not args.get("force"):
        return {"status": "exists"}
    snapshot_download(args.get("repo", BASE_MODEL), local_dir=dst)
    vol.commit()
    return {"status": "downloaded", "files": sorted(os.listdir(dst))}


@app.function(image=cpu_image, volumes=MOUNTS, timeout=6 * 3600)
def build_corpus(job_id: str, args: dict):
    """Fetch every URL in the manifest and check its text hash against the one we annotated."""
    import concurrent.futures as cf

    from owl.scrape import Fetcher, fetch_doc

    manifest = [json.loads(l) for l in open(_resolve(args.get("manifest", "data/manifest.jsonl"))) if l.strip()]
    fetcher = Fetcher()
    stats = {"total": len(manifest), "ok": 0, "hash_mismatch": 0, "missing": 0, "by_source": {}}
    out = Path(VOL, "corpus.jsonl")
    with open(out, "w") as f, cf.ThreadPoolExecutor(8) as ex:
        for m, doc in zip(manifest, ex.map(lambda m: fetch_doc(fetcher, m["source"], m["url"]), manifest)):
            if doc is None:
                stats["missing"] += 1
                continue
            if doc["sha256"] != m["sha256"]:
                # Paragraph indices may have shifted; keep it, but flag it so units can be checked.
                stats["hash_mismatch"] += 1
                doc["hash_mismatch"] = True
            else:
                stats["ok"] += 1
            stats["by_source"][doc["source"]] = stats["by_source"].get(doc["source"], 0) + 1
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    vol.commit()
    (_results_dir(job_id) / "corpus_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


@app.function(image=cpu_image, volumes=MOUNTS, timeout=3600)
def build_sft(job_id: str, args: dict):
    """Join annotator units with the corpus; optionally mix in retention examples."""
    import random

    from owl.units import load_units, materialize

    corpus = {d["doc_id"]: d for d in map(json.loads, open(f"{VOL}/corpus.jsonl"))}
    units = load_units(_resolve(args.get("units", "data/backtranslations")))
    holdout = set(json.load(open(_resolve(args["holdout"])))) if args.get("holdout") else set()
    units = [u for u in units if u.get("doc_id") not in holdout]
    examples, errors = materialize(units, corpus, max_words=args.get("max_words", 4500))
    bad = {d for d, doc in corpus.items() if doc.get("hash_mismatch")}
    examples = [e for e in examples if e["doc_id"] not in bad or args.get("allow_mismatch")]
    # Source weights from configs/sources.yaml: upsample by integer part + random fraction.
    import yaml

    weights = {s["name"]: s.get("weight", 1.0) for s in yaml.safe_load(open(f"{REPO}/configs/sources.yaml"))["sources"]}
    rng = random.Random(0)
    mixed = []
    for e in examples:
        w = weights.get(e["source"], 1.0)
        k = int(w) + (1 if rng.random() < w - int(w) else 0)
        mixed += [e] * k
    if args.get("retention"):
        ret = [json.loads(l) for l in open(_resolve(args["retention"]))]
        n = int(len(mixed) * args.get("retention_frac", 0.25))
        rng.shuffle(ret)
        mixed += [{"id": r["id"], "domain": "retention", "messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]} for r in ret[:n]]
    rng.shuffle(mixed)
    out = Path(VOL, "data", args.get("out", "sft.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for e in mixed:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    vol.commit()
    from collections import Counter

    stats = {
        "units": len(units),
        "examples": len(examples),
        "after_weighting": len(mixed),
        "errors": errors[:200],
        "n_errors": len(errors),
        "by_domain": Counter(e["domain"] for e in mixed),
        "by_source": Counter(e.get("source", "retention") for e in mixed),
        "mean_words": sum(len(e["messages"][-1]["content"].split()) for e in mixed) / max(len(mixed), 1),
    }
    (_results_dir(job_id) / f"sft_stats_{out.stem}.json").write_text(json.dumps(stats, indent=1))
    # Export the prompts only (never the human text) so they can be reviewed in the repo.
    with open(_results_dir(job_id) / f"sft_prompts_{out.stem}.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps({"id": e["id"], "domain": e["domain"], "prompt": e["messages"][0]["content"]}, ensure_ascii=False) + "\n")
    return stats


@app.function(image=cpu_image, volumes=MOUNTS, timeout=3600)
def build_human_pairs(job_id: str, args: dict):
    """Offline DPO pairs: chosen = human text, rejected = base model's answer to the same prompt."""
    sft = {e["id"]: e for e in map(json.loads, open(_resolve(args["sft"]))) if e.get("domain") != "retention"}
    samples = [json.loads(l) for l in open(_resolve(args["samples"]))]
    pairs = []
    for s in samples:
        e = sft.get(s["id"])
        if not e or s.get("finish_reason") != "stop":
            continue
        pairs.append({"id": s["id"], "prompt": e["messages"][0]["content"], "chosen": e["messages"][-1]["content"], "rejected": s["response"]})
    out = Path(VOL, "data", args.get("out", "human_pairs.jsonl"))
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    vol.commit()
    return {"pairs": len(pairs)}


@app.function(image=vllm_image, gpu=os.environ.get("OWL_GEN_GPU", "H200"), volumes=MOUNTS, timeout=6 * 3600)
def generate(job_id: str, args: dict):
    vol.reload()
    rd = _results_dir(job_id)
    out = rd / args["out"]
    cmd = [
        sys.executable, "-m", "owl.generate",
        "--model", _model_path(args.get("model", "base")),
        "--prompts", _resolve(args["prompts"]),
        "--out", str(out),
        "--tag", args.get("tag", args.get("model", "base")),
        "--n", str(args.get("n", 1)),
        "--temperature", str(args.get("temperature", 0.8)),
        "--max-tokens", str(args.get("max_tokens", 4096)),
        "--tp", str(args.get("tp", 1)),
    ]
    if args.get("limit"):
        cmd += ["--limit", str(args["limit"])]
    _run(cmd, rd / "generate.log")
    if args.get("private"):  # e.g. samples that will be paired with human text: keep off the repo
        Path(VOL, "data").mkdir(exist_ok=True)
        out.rename(Path(VOL, "data", args["out"]))
    vol.commit()
    from owl.slop import aggregate

    rows = [json.loads(l) for l in open(out if out.exists() else Path(VOL, "data", args["out"]))]
    rep = {"rows": len(rows), "truncated": sum(r["finish_reason"] != "stop" for r in rows), "slop": aggregate([r["response"] for r in rows])}
    (rd / f"{Path(args['out']).stem}_summary.json").write_text(json.dumps(rep, indent=1))
    vol.commit()
    return rep


@app.function(image=train_image, gpu=os.environ.get("OWL_TRAIN_GPU", "H200"), volumes=MOUNTS, timeout=24 * 3600, memory=200 * 1024)
def train(job_id: str, args: dict):
    """args: {stage: sft|dpo, model, data, out_model, hp: {...}}. Trains LoRA, then merges."""
    vol.reload()
    rd = _results_dir(job_id)
    adapter = f"{VOL}/adapters/{args['out_model']}"
    cmd = [
        sys.executable, "-m", "owl.train", args["stage"],
        "--model", _model_path(args.get("model", "base")),
        "--data", _resolve(args["data"]),
        "--out", adapter,
    ]
    for k, v in (args.get("hp") or {}).items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    _run(cmd, rd / f"train_{args['out_model']}.log")
    vol.commit()
    _run([sys.executable, "-m", "owl.train", "merge", "--model", _model_path(args.get("model", "base")), "--adapter", adapter, "--out", _model_path(args["out_model"])], rd / f"merge_{args['out_model']}.log")
    import shutil

    shutil.copy(f"{adapter}/log_history.json", rd / f"log_history_{args['out_model']}.json")
    vol.commit()
    return {"model": args["out_model"]}


@app.function(image=cpu_image, volumes=MOUNTS, timeout=600)
def ls(job_id: str, args: dict):
    out = {}
    for sub in ("models", "data", "adapters", "results"):
        p = Path(VOL, sub)
        out[sub] = sorted(os.listdir(p)) if p.exists() else []
    out["corpus_lines"] = sum(1 for _ in open(f"{VOL}/corpus.jsonl")) if Path(VOL, "corpus.jsonl").exists() else 0
    (_results_dir(job_id) / "ls.json").write_text(json.dumps(out, indent=1))
    vol.commit()
    return out


STEPS = {
    "download_base": download_base,
    "build_corpus": build_corpus,
    "build_sft": build_sft,
    "build_human_pairs": build_human_pairs,
    "generate": generate,
    "train": train,
    "ls": ls,
}


@app.local_entrypoint()
def main(spec: str):
    import yaml

    job = yaml.safe_load(open(spec))
    job_id = job["id"]
    report = {"id": job_id, "steps": []}
    # Independent steps can be grouped: a list inside `steps` runs in parallel.
    for step in job["steps"]:
        group = step if isinstance(step, list) else [step]
        calls = []
        for s in group:
            (name, args), = s.items()
            print(f"[job {job_id}] start {name} {json.dumps(args or {})}", flush=True)
            calls.append((name, STEPS[name].spawn(job_id, args or {})))
        for name, call in calls:
            try:
                res = call.get()
                report["steps"].append({"step": name, "ok": True, "result": res})
                print(f"[job {job_id}] done {name}: {json.dumps(res, default=str)[:2000]}", flush=True)
            except Exception as e:
                report["steps"].append({"step": name, "ok": False, "error": repr(e)[:4000]})
                print(f"[job {job_id}] FAILED {name}: {e!r}", flush=True)
                Path("job_report.json").write_text(json.dumps(report, indent=1, default=str))
                raise
    Path("job_report.json").write_text(json.dumps(report, indent=1, default=str))
