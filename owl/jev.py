"""Judge comparisons with Jev (TypeSafe's decision model on OpenRouter) instead of agents.

Reads DIR/comparisons.jsonl (from owl.judging prep) and writes DIR/verdicts/jev.jsonl in
the agents' verdict format, so `owl.judging collect` works unchanged. Jev answers a
choice question with probabilities; p(A) maps onto the verdict scale.

    python -m owl.jev --dir DIR [--workers 16] [--limit N]

The key comes from OPENROUTERAPIKEY (any value works where a proxy injects the real key).
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from owl.rubric import RUBRIC

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.14"
CRITERIA = {x: f"Response {x} is the one a discerning reader, who loves good essays and has "
    "no patience for filler, would rather have received: right and well-developed ideas in a "
    "real train of thought, natural varied rhythm, concrete, honest, correct, and shaped to "
    "the request. It has less of what the rubric counts against: headers, bullets and bold "
    "applied to prose, stock phrases, filler openings and summaries, hedging, padding, "
    f"errors, sycophancy. Longer or more formatted is not better." for x in "AB"}


def verdict(p):
    return ("A>>B" if p >= 0.9 else "A>B" if p >= 0.6 else "A=B" if p > 0.4
        else "B>A" if p > 0.1 else "B>>A")


def judge(client, c):
    state = {"rubric": RUBRIC, "request": c["prompt"], "response_A": c["a"],
        "response_B": c["b"]}
    body = {"model": MODEL, "state": state,
        "questions": {"better": {"type": "choice", "criteria": CRITERIA}}}
    for attempt in range(4):
        r = client.post(URL, json=body)
        if r.status_code == 200:
            break
    r.raise_for_status()
    d = r.json()
    p = d["answers"]["better"]["probabilities"]["A"]
    return {"cid": c["cid"], "p_a": p, "verdict": verdict(p), "judge": d["model"],
        "cost": d["usage"]["cost"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    d = Path(a.dir)
    comps = [json.loads(l) for l in open(d / "comparisons.jsonl")][:a.limit]
    out = d / "verdicts" / "jev.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["cid"] for l in open(out)} if out.exists() else set()
    todo = [c for c in comps if c["cid"] not in done]
    key = os.environ.get("OPENROUTERAPIKEY", "proxy-injected")
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120)
    cost = 0.0
    with open(out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for v in ex.map(lambda c: judge(client, c), todo):
            f.write(json.dumps(v) + "\n")
            cost += v["cost"]
    print(json.dumps({"judged": len(todo), "skipped": len(done), "cost_usd": round(cost, 4)}))


if __name__ == "__main__":
    main()
