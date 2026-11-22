# Build round-3 pairs from 016a samples. Usage: python build_r3.py [verdict_dir]
import json, re, sys, collections, statistics as st
R = "ops/results/016c-r3-candidates"
style = open("configs/style_system_prompt.txt").read()
import glob
think = {r["id"]: r for f in sorted(glob.glob(f"{R}/r3_think_*.jsonl")) for r in map(json.loads, open(f))}
plain = {r["id"]: r for r in map(json.loads, open(f"{R}/r3_plain.jsonl"))}
def loops(t):
    ps = [p.strip() for p in t.split("\n\n") if len(p.split()) >= 8]
    return len(ps) != len(set(ps))
why = collections.Counter(); keep = []
for i, t in think.items():
    p = plain.get(i)
    if t["think_cut"]: why["think_cut"] += 1; continue
    if t["finish_reason"] != "stop" or not t["response"].strip(): why["unfinished"] += 1; continue
    if loops(t["response"]): why["loop"] += 1; continue
    if not p or p["finish_reason"] != "stop": why["plain_unfinished"] += 1; continue
    keep.append({"id": i, "domain": t["domain"], "prompt": t["prompt"], "system": style,
        "chosen": t["response"], "rejected": p["response"]})
print("kept", len(keep), dict(why), collections.Counter(k["domain"] for k in keep))
ratio = [len(k["chosen"].split()) / max(len(k["rejected"].split()), 1) for k in keep]
print("chosen/rejected word ratio median", round(st.median(ratio), 2), ">2.5x:", sum(r > 2.5 for r in ratio))
json.dump(keep, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"))
