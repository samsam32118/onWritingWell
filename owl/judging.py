"""Pairwise judging by Claude subagents: prepare chunks, collect verdicts.

    1. prep:    samples -> comparisons -> chunk files (rubric + ~30 comparisons each)
    2. agents:  ops/workflows/judge.js, one subagent per chunk, writes verdicts JSONL
    3. collect: verdicts -> win rates (vs) or DPO pairs (tournament)

Position bias: every comparison randomises which response is A, and in `vs` mode each pair
is judged in both orders and averaged, so a judge that always picks A scores a tie.

    python -m owl.judging prep vs --a REF.jsonl --b NEW.jsonl --out DIR
    python -m owl.judging prep tournament --samples S.jsonl --out DIR [--k 4]
    python -m owl.judging collect vs --dir DIR
    python -m owl.judging collect tournament --dir DIR --pairs-out pairs.jsonl
"""
import argparse
import hashlib
import itertools
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from owl.rubric import JUDGE_TASK, RUBRIC, RUBRIC_VERSION, VERDICT_SCALE


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def rng(key):
    return random.Random(int(hashlib.sha1(key.encode()).hexdigest()[:8], 16))


def keyed(r):
    return {**r, "_key": f"{r['model_tag']}#{r['id']}#{r.get('sample_idx', 0)}"}


def comparison(cid, prompt, left, right):
    return {"cid": cid, "prompt": prompt, "a": left["response"], "b": right["response"],
        "a_key": left["_key"], "b_key": right["_key"]}


def prep_vs(ref_path, new_path, out_dir):
    """Reference vs challenger, first sample of each, judged in both orders."""
    ref = {r["id"]: keyed(r) for r in load(ref_path) if r.get("sample_idx", 0) == 0}
    new = {r["id"]: keyed(r) for r in load(new_path) if r.get("sample_idx", 0) == 0}
    comps = []
    for pid in sorted(ref.keys() & new.keys()):
        comps.append(comparison(f"{pid}~0", ref[pid]["prompt"], ref[pid], new[pid]))
        comps.append(comparison(f"{pid}~1", ref[pid]["prompt"], new[pid], ref[pid]))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tags = [next(iter(ref.values()))["model_tag"], next(iter(new.values()))["model_tag"]]
    Path(out_dir, "tags.json").write_text(json.dumps(tags))
    return comps


def prep_tournament(samples_path, k=4):
    """Round robin among the first k untruncated samples of each prompt, one random order
    per pair. Truncated answers lose automatically, so no judging is spent on them."""
    by_id = defaultdict(list)
    for r in load(samples_path):
        if r.get("finish_reason", "stop") == "stop":
            by_id[r["id"]].append(keyed(r))
    comps = []
    for pid, rs in sorted(by_id.items()):
        g = rng(pid)
        pairs = list(itertools.combinations(rs[:k], 2))
        g.shuffle(pairs)
        for j, (x, y) in enumerate(pairs):
            l, r = (x, y) if g.random() < 0.5 else (y, x)
            comps.append(comparison(f"{pid}~{j}", x["prompt"], l, r))
    return comps


def write_chunks(comps, out_dir, chunk=30):
    d = Path(out_dir)
    (d / "chunks").mkdir(parents=True, exist_ok=True)
    (d / "verdicts").mkdir(exist_ok=True)
    (d / "comparisons.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in comps))
    task = JUDGE_TASK.rsplit("\nCompare A and B", 1)[0]
    chunks = []
    for i in range(0, len(comps), chunk):
        part = comps[i:i + chunk]
        name = f"chunk-{i // chunk:04d}"
        (d / "chunks" / f"{name}.txt").write_text(
            f"RUBRIC ({RUBRIC_VERSION})\n\n{RUBRIC}\n\n" + "=" * 80 + "".join(
                f"\n\n######## COMPARISON cid={c['cid']}\n\n"
                + task.format(prompt=c["prompt"], a=c["a"], b=c["b"]) for c in part))
        chunks.append({"name": name, "n": len(part)})
    (d / "workflow_args.json").write_text(json.dumps({"dir": str(d.resolve()),
        "chunks": chunks}))
    return chunks


def verdicts(d):
    """Valid verdicts by cid; malformed lines from an agent are ignored and re-judged."""
    out = {}
    for p in sorted(Path(d, "verdicts").glob("*.jsonl")):
        for line in open(p):
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(v, dict) and v.get("verdict") in VERDICT_SCALE:
                out[v["cid"]] = v
    return out


def collect_vs(out_dir):
    """Win rate of the challenger (second tag) against the reference, overall and by the
    domain prefix of the prompt id."""
    d = Path(out_dir)
    challenger = json.loads((d / "tags.json").read_text())[1]
    comps = load(d / "comparisons.jsonl")
    vs = verdicts(d)
    per_pid = defaultdict(list)
    for c in comps:
        if c["cid"] in vs:
            s = VERDICT_SCALE[vs[c["cid"]]["verdict"]]
            per_pid[c["cid"].split("~")[0]].append(
                s if c["a_key"].startswith(challenger + "#") else 1 - s)
    scores = {pid: statistics.fmean(ss) for pid, ss in per_pid.items()}
    both = [ss for ss in per_pid.values() if len(ss) == 2]
    by_dom = defaultdict(list)
    for pid, s in scores.items():
        by_dom[pid.split("-")[0]].append(s)
    return {"challenger": challenger, "judged_prompts": len(scores),
        "win_rate": round(statistics.fmean(scores.values()), 4) if scores else None,
        "strict_wins": sum(s > 0.5 for s in scores.values()),
        "ties": sum(s == 0.5 for s in scores.values()),
        "losses": sum(s < 0.5 for s in scores.values()),
        # Both orders agree on direction: a measure of how much position swayed the judge.
        "order_consistency": round(sum((a - .5) * (b - .5) > 0 for a, b in both)
            / max(len(both), 1), 3),
        "missing": len(comps) - sum(c["cid"] in vs for c in comps),
        "by_domain": {k: {"n": len(v), "win_rate": round(statistics.fmean(v), 4)}
            for k, v in sorted(by_dom.items())}}


def collect_tournament(out_dir, pairs_out, min_margin=0.34):
    """Score each sample by mean preference; emit (best, worst) pairs with a clear margin."""
    d = Path(out_dir)
    comps = load(d / "comparisons.jsonl")
    vs = verdicts(d)
    pts, text, prompts = defaultdict(list), {}, {}
    for c in comps:
        pid = c["cid"].split("~")[0]
        text[c["a_key"]], text[c["b_key"]], prompts[pid] = c["a"], c["b"], c["prompt"]
        if c["cid"] in vs:
            s = VERDICT_SCALE[vs[c["cid"]]["verdict"]]
            pts[pid, c["a_key"]].append(s)
            pts[pid, c["b_key"]].append(1 - s)
    by_pid = defaultdict(list)
    for (pid, key), ss in pts.items():
        by_pid[pid].append((statistics.fmean(ss), key))
    pairs = []
    for pid, scored in by_pid.items():
        (lo, lk), (hi, hk) = min(scored), max(scored)
        if hi - lo >= min_margin:
            pairs.append({"id": pid, "prompt": prompts[pid], "chosen": text[hk],
                "rejected": text[lk], "margin": round(hi - lo, 3)})
    Path(pairs_out).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs))
    lc = [len(p["chosen"].split()) for p in pairs]
    lr = [len(p["rejected"].split()) for p in pairs]
    return {"prompts_scored": len(by_pid), "pairs": len(pairs),
        "missing": len(comps) - sum(c["cid"] in vs for c in comps),
        "chosen_words_mean": round(statistics.fmean(lc), 1) if lc else None,
        "rejected_words_mean": round(statistics.fmean(lr), 1) if lr else None,
        # If chosen is systematically longer, the judge may be rewarding length.
        "chosen_longer_frac": round(sum(c > r for c, r in zip(lc, lr)) / max(len(pairs), 1), 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "collect"])
    ap.add_argument("mode", choices=["vs", "tournament"])
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--samples")
    ap.add_argument("--out")
    ap.add_argument("--dir")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--pairs-out")
    ap.add_argument("--min-margin", type=float, default=0.34)
    a = ap.parse_args()
    if a.cmd == "prep":
        comps = prep_vs(a.a, a.b, a.out) if a.mode == "vs" else prep_tournament(a.samples, a.k)
        chunks = write_chunks(comps, a.out, a.chunk)
        print(json.dumps({"comparisons": len(comps), "chunks": len(chunks)}))
    elif a.mode == "vs":
        print(json.dumps(collect_vs(a.dir), indent=1))
    else:
        print(json.dumps(collect_tournament(a.dir, a.pairs_out, a.min_margin), indent=1))
