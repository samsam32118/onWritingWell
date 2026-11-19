"""Validate annotator units against the local corpus and write the committed manifest.

    python scripts/build_manifest.py CORPUS.jsonl [--units data/backtranslations]

Writes
    data/manifest.jsonl     {doc_id, source, url, title, words, sha256} for every annotated doc
    data/holdout.json       doc_ids held out of training (5% by hash) for human-reference evals
    data/eval/human_ref_prompts.jsonl   prompts of held-out essay units (text stays off-repo)
Prints a validation report; units that fail validation are listed so they can be fixed.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from owl.units import load_units, span_text, validate_unit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("corpus")
ap.add_argument("--units", default="data/backtranslations")
ap.add_argument("--holdout-mod", type=int, default=20)
a = ap.parse_args()

corpus = {d["doc_id"]: d for d in map(json.loads, open(a.corpus))}
units = load_units(a.units)
errors, ok = [], []
for u in units:
    if u.get("skip"):
        continue
    err = validate_unit(u, corpus.get(u.get("doc_id")))
    (errors.append((u, err)) if err else ok.append(u))

doc_ids = sorted({u["doc_id"] for u in ok})
holdout = sorted(d for d in doc_ids if int(hashlib.sha1(d.encode()).hexdigest(), 16) % a.holdout_mod == 0)
Path("data").mkdir(exist_ok=True)
with open("data/manifest.jsonl", "w") as f:
    for d in doc_ids:
        doc = corpus[d]
        f.write(json.dumps({k: doc[k] for k in ("doc_id", "source", "url", "title", "words", "sha256")}, ensure_ascii=False) + "\n")
Path("data/holdout.json").write_text(json.dumps(holdout, indent=0))
Path("data/eval").mkdir(parents=True, exist_ok=True)
with open("data/eval/human_ref_prompts.jsonl", "w") as f:
    for u in ok:
        if u["doc_id"] in holdout and u["kind"] == "essay":
            f.write(json.dumps({"id": f"href-{u['doc_id']}:{u['para_start']}-{u['para_end']}", "domain": u["domain"], "prompt": u["prompt"]}, ensure_ascii=False) + "\n")

skips = [u for u in units if u.get("skip")]
words = Counter()
for u in ok:
    words[u["kind"]] += len(span_text(corpus[u["doc_id"]], u["para_start"], u["para_end"]).split())
print(json.dumps({
    "docs_annotated": len({u["doc_id"] for u in units}),
    "docs_skipped": len(skips),
    "units_ok": len(ok),
    "units_bad": len(errors),
    "by_kind": Counter(u["kind"] for u in ok),
    "by_domain": Counter(u["domain"] for u in ok),
    "by_source": Counter(corpus[u["doc_id"]]["source"] for u in ok),
    "words_by_kind": words,
    "holdout_docs": len(holdout),
    "skip_reasons_sample": [s.get("reason", "")[:80] for s in skips[:15]],
}, indent=1, default=dict))
for u, e in errors[:40]:
    print("BAD", u.get("doc_id"), u.get("para_start"), u.get("para_end"), e)
