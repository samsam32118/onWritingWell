"""Write each scraped doc as a numbered-paragraph text file for the annotator agents.

    python scripts/prep_annotation.py CORPUS.jsonl OUTDIR [--skip-done data/backtranslations]
Prints one batch per line: comma-separated doc_ids, grouped to ~BATCH_WORDS words.
"""
import json, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from owl.units import numbered, load_units

ap = argparse.ArgumentParser()
ap.add_argument("corpus"); ap.add_argument("outdir")
ap.add_argument("--skip-done", default=None)
ap.add_argument("--batch-words", type=int, default=22000)
ap.add_argument("--only", default="")
a = ap.parse_args()
done = {u["doc_id"] for u in load_units(a.skip_done)} if a.skip_done and Path(a.skip_done).exists() else set()
only = set(filter(None, a.only.split(",")))
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
docs = [json.loads(l) for l in open(a.corpus)]
docs = [d for d in docs if d["doc_id"] not in done and (not only or d["source"] in only)]
batches, cur, words = [], [], 0
for d in docs:
    (out / f"{d['doc_id']}.txt").write_text(numbered(d))
    if cur and words + d["words"] > a.batch_words:
        batches.append(cur); cur, words = [], 0
    cur.append(d["doc_id"]); words += d["words"]
if cur: batches.append(cur)
for b in batches: print(",".join(b))
print(f"{len(docs)} docs, {len(batches)} batches", file=sys.stderr)
