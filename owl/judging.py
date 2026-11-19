"""Pairwise judging by Claude subagents (no API key): prepare chunks, collect verdicts.

Flow
    1. prep:    samples -> comparisons -> chunk files (rubric + ~N comparisons each)
    2. agents:  one subagent per chunk writes verdicts JSONL {cid, verdict, reason}
    3. collect: verdicts -> win rates (eval) or DPO pairs (training rounds)

Position bias: every comparison randomises which response is A. In `vs` mode each pair
is judged in both orders and the two verdicts are averaged; a pair whose two verdicts
disagree in direction is a tie.

    python -m owl.judging prep vs --a FILE --b FILE --out DIR [--chunk 30]
    python -m owl.judging prep tournament --samples FILE --out DIR [--k 4]
    python -m owl.judging collect vs --dir DIR
    python -m owl.judging collect tournament --dir DIR --pairs-out pairs.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from owl.rubric import JUDGE_TASK, RUBRIC, RUBRIC_VERSION, VERDICT_SCALE


def _load(path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _rng(*keys) -> random.Random:
    return random.Random(int(hashlib.sha1("|".join(map(str, keys)).encode()).hexdigest()[:8], 16))


def _comparison(cid: str, prompt: str, left: dict, right: dict) -> dict:
    return {"cid": cid, "prompt": prompt, "a": left["response"], "b": right["response"], "a_key": left["_key"], "b_key": right["_key"]}


def prep_vs(a_path: str, b_path: str, both_orders: bool = True) -> list[dict]:
    """Compare model A vs model B, one sample each per prompt id."""
    A = {r["id"]: {**r, "_key": f"{r['model_tag']}#{r['id']}#{r.get('sample_idx', 0)}"} for r in _load(a_path) if r.get("sample_idx", 0) == 0}
    B = {r["id"]: {**r, "_key": f"{r['model_tag']}#{r['id']}#{r.get('sample_idx', 0)}"} for r in _load(b_path) if r.get("sample_idx", 0) == 0}
    out = []
    for pid in sorted(A.keys() & B.keys()):
        a, b = A[pid], B[pid]
        orders = [(a, b), (b, a)] if both_orders else [(a, b) if _rng(pid).random() < 0.5 else (b, a)]
        for j, (l, r) in enumerate(orders):
            out.append(_comparison(f"{pid}~{j}", a["prompt"], l, r))
    return out


def prep_tournament(samples_path: str, k: int = 4, max_pairs: int | None = None) -> list[dict]:
    """Round robin among the k samples of each prompt (one random order per pair)."""
    by_id = defaultdict(list)
    for r in _load(samples_path):
        if r.get("finish_reason", "stop") != "stop":
            continue  # truncated answers lose automatically; don't spend judging on them
        by_id[r["id"]].append({**r, "_key": f"{r['model_tag']}#{r['id']}#{r['sample_idx']}"})
    out = []
    for pid, rs in sorted(by_id.items()):
        rs = rs[:k]
        pairs = list(itertools.combinations(rs, 2))
        rng = _rng(pid)
        rng.shuffle(pairs)
        for j, (x, y) in enumerate(pairs[:max_pairs]):
            l, r = (x, y) if rng.random() < 0.5 else (y, x)
            out.append(_comparison(f"{pid}~{j}", x["prompt"], l, r))
    return out


def write_chunks(comps: list[dict], out_dir: str, chunk: int = 30) -> list[str]:
    d = Path(out_dir)
    (d / "chunks").mkdir(parents=True, exist_ok=True)
    (d / "verdicts").mkdir(exist_ok=True)
    with open(d / "comparisons.jsonl", "w") as f:
        for c in comps:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    paths = []
    for i in range(0, len(comps), chunk):
        part = comps[i : i + chunk]
        body = [f"RUBRIC ({RUBRIC_VERSION})\n\n{RUBRIC}\n\n" + "=" * 80]
        for c in part:
            body.append(f"\n\n######## COMPARISON cid={c['cid']}\n\n" + JUDGE_TASK.format(prompt=c["prompt"], a=c["a"], b=c["b"]).rsplit("\nCompare A and B", 1)[0])
        p = d / "chunks" / f"chunk-{i // chunk:04d}.txt"
        p.write_text("\n".join(body))
        paths.append(str(p))
    # Args for ops/workflows/judge.js
    (d / "workflow_args.json").write_text(json.dumps({"dir": str(d.resolve()), "chunks": [{"name": Path(p).stem, "n": min(chunk, len(comps) - i * chunk)} for i, p in enumerate(paths)]}))
    return paths


def _verdicts(d: Path) -> dict[str, dict]:
    out = {}
    for p in sorted((d / "verdicts").glob("*.jsonl")):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            if v.get("verdict") in VERDICT_SCALE:
                out[v["cid"]] = v
    return out


def missing(out_dir: str) -> list[str]:
    d = Path(out_dir)
    have = _verdicts(d)
    return [c["cid"] for c in _load(d / "comparisons.jsonl") if c["cid"] not in have]


def collect_vs(out_dir: str) -> dict:
    """Win rate of the model in file --b (challenger) against --a (reference), per domain."""
    d = Path(out_dir)
    comps = {c["cid"]: c for c in _load(d / "comparisons.jsonl")}
    vs = _verdicts(d)
    per_pair = defaultdict(list)
    tags = set()
    for cid, c in comps.items():
        if cid not in vs:
            continue
        s = VERDICT_SCALE[vs[cid]["verdict"]]  # score for whichever response was shown as A
        a_tag = c["a_key"].split("#")[0]
        tags |= {a_tag, c["b_key"].split("#")[0]}
        per_pair[cid.split("~")[0]].append((a_tag, s))
    tags = sorted(tags)
    ref = _load_tag_order(d) or tags
    challenger = ref[1] if len(ref) > 1 else tags[-1]
    scores, consistent = {}, 0
    for pid, items in per_pair.items():
        ss = [s if t == challenger else 1 - s for t, s in items]
        scores[pid] = statistics.fmean(ss)
        if len(ss) == 2 and (ss[0] - 0.5) * (ss[1] - 0.5) >= 0:
            consistent += 1
    by_dom = defaultdict(list)
    for pid, s in scores.items():
        by_dom[pid.split("-")[0] if "-" in pid else "all"].append(s)
    return {
        "challenger": challenger,
        "judged_prompts": len(scores),
        "win_rate": round(statistics.fmean(scores.values()), 4) if scores else None,
        "strict_wins": sum(s > 0.5 for s in scores.values()),
        "ties": sum(s == 0.5 for s in scores.values()),
        "losses": sum(s < 0.5 for s in scores.values()),
        "order_consistency": round(consistent / max(sum(len(v) == 2 for v in per_pair.values()), 1), 3),
        "missing": len(comps) - len(vs),
        "by_domain": {k: {"n": len(v), "win_rate": round(statistics.fmean(v), 4)} for k, v in sorted(by_dom.items())},
    }


def _load_tag_order(d: Path):
    p = d / "tags.json"
    return json.loads(p.read_text()) if p.exists() else None


def collect_tournament(out_dir: str, pairs_out: str | None, min_margin: float = 0.34) -> dict:
    """Score each sample by mean preference; emit (best, worst) pairs with a clear margin."""
    d = Path(out_dir)
    comps = _load(d / "comparisons.jsonl")
    vs = _verdicts(d)
    pts = defaultdict(list)
    text = {}
    prompts = {}
    for c in comps:
        v = vs.get(c["cid"])
        text[c["a_key"]], text[c["b_key"]] = c["a"], c["b"]
        pid = c["cid"].split("~")[0]
        prompts[pid] = c["prompt"]
        if not v:
            continue
        s = VERDICT_SCALE[v["verdict"]]
        pts[(pid, c["a_key"])].append(s)
        pts[(pid, c["b_key"])].append(1 - s)
    by_pid = defaultdict(list)
    for (pid, key), ss in pts.items():
        by_pid[pid].append((statistics.fmean(ss), key))
    pairs = []
    for pid, scored in by_pid.items():
        scored.sort()
        (lo, lk), (hi, hk) = scored[0], scored[-1]
        if hi - lo >= min_margin:
            pairs.append({"id": pid, "prompt": prompts[pid], "chosen": text[hk], "rejected": text[lk], "margin": round(hi - lo, 3)})
    if pairs_out:
        with open(pairs_out, "w") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    lens_c = [len(p["chosen"].split()) for p in pairs]
    lens_r = [len(p["rejected"].split()) for p in pairs]
    return {
        "prompts_scored": len(by_pid),
        "pairs": len(pairs),
        "missing_verdicts": len(comps) - sum(c["cid"] in vs for c in comps),
        "chosen_words_mean": round(statistics.fmean(lens_c), 1) if lens_c else None,
        "rejected_words_mean": round(statistics.fmean(lens_r), 1) if lens_r else None,
        # If chosen is systematically longer, the judge may be rewarding length: watch this.
        "chosen_longer_frac": round(sum(c > r for c, r in zip(lens_c, lens_r)) / max(len(pairs), 1), 3),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "collect", "missing"])
    ap.add_argument("mode", choices=["vs", "tournament"], nargs="?")
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--samples")
    ap.add_argument("--out"); ap.add_argument("--dir")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--pairs-out")
    ap.add_argument("--min-margin", type=float, default=0.34)
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        comps = prep_vs(a.a, a.b) if a.mode == "vs" else prep_tournament(a.samples, a.k, a.max_pairs)
        paths = write_chunks(comps, a.out, a.chunk)
        if a.mode == "vs":
            ta, tb = _load(a.a)[0]["model_tag"], _load(a.b)[0]["model_tag"]
            Path(a.out, "tags.json").write_text(json.dumps([ta, tb]))
        print(json.dumps({"comparisons": len(comps), "chunks": len(paths)}))
    elif a.cmd == "missing":
        print(json.dumps(missing(a.dir)))
    else:
        res = collect_vs(a.dir) if a.mode == "vs" else collect_tournament(a.dir, a.pairs_out, a.min_margin)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
