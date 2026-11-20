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

import re

from owl.rubric import JUDGE_TASK, RUBRIC, VERDICT_SCALE

URL = "https://openrouter.ai/api/alpha/decisions"
CHAT = "https://openrouter.ai/api/v1/chat/completions"
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


def judge(client, c, model):
    if model != MODEL:
        return judge_chat(client, c, model)
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


def judge_chat(client, c, model):
    """Any OpenRouter chat model, given the same rubric and task text as the agents."""
    msg = f"{RUBRIC}\n\n" + JUDGE_TASK.format(prompt=c["prompt"], a=c["a"], b=c["b"]) + (
        "\nEnd with a final line of the form: VERDICT: <A>>B|A>B|A=B|B>A|B>>A>")
    body = {"model": model, "messages": [{"role": "user", "content": msg}],
        "temperature": 0, "max_tokens": 8000, "usage": {"include": True}}
    for attempt in range(4):
        r = client.post(CHAT, json=body)
        try:
            d = r.json()
            text = d["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError):
            continue
        m = re.findall(r"VERDICT:\s*\**\s*(A>>B|A>B|A=B|B>A|B>>A)", text)
        if m:
            return {"cid": c["cid"], "verdict": m[-1], "reason": text[-600:],
                "judge": d.get("model", model), "cost": d.get("usage", {}).get("cost", 0)}
    return {"cid": c["cid"], "verdict": None, "judge": model, "cost": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="jev")
    a = ap.parse_args()
    d = Path(a.dir)
    comps = [json.loads(l) for l in open(d / "comparisons.jsonl")][:a.limit]
    out = d / "verdicts" / f"{a.out}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {v["cid"] for v in map(json.loads, open(out)) if v["verdict"]} if out.exists() else set()
    todo = [c for c in comps if c["cid"] not in done]
    key = os.environ.get("OPENROUTERAPIKEY", "proxy-injected")
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=600)
    cost = 0.0
    with open(out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for v in ex.map(lambda c: judge(client, c, a.model), todo):
            f.write(json.dumps(v) + "\n")
            cost += v["cost"]
    print(json.dumps({"judged": len(todo), "skipped": len(done), "cost_usd": round(cost, 4)}))


if __name__ == "__main__":
    main()
