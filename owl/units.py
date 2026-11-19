"""Training units: (prompt, human-written span of a document).

Spans are paragraph index ranges, so the repo can commit prompts and spans without the
copyrighted text. data/backtranslations/*.jsonl, written by the annotator agents, holds
    {doc_id, kind: essay|passage, domain, para_start, para_end, prompt}
    or {doc_id, skip: true, reason}
Modal joins them with the corpus in modal_app.build_sft.

    python -m owl.units prep CORPUS OUTDIR [--only src,...]   numbered docs + batch lists
    python -m owl.units manifest CORPUS                      validate units, write manifest
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

NOTES_HEADING = re.compile(r"^(notes|footnotes|references|thanks|acknowledg(e)?ments)\s*:?$",
    re.I)


def paragraphs(text):
    return [p for p in text.split("\n\n") if p.strip()]


def span_text(doc, start, end):
    """Paragraphs start..end inclusive, stopping at a Notes/Thanks heading, with footnote
    markers ("[1]", "[2, 3]") removed: they belong on a web page, not in a chat reply."""
    out = []
    for p in paragraphs(doc["text"])[start:end + 1]:
        if NOTES_HEADING.match(p.strip()):
            break
        out.append(p)
    return re.sub(r"\s?\[\d+(,\s*\d+)*\]", "", "\n\n".join(out).strip())


def load_units(directory):
    return [json.loads(l) for p in sorted(Path(directory).glob("*.jsonl"))
        for l in open(p) if l.strip()]


def validate_unit(u, doc):
    """Error string for a unit an annotator got wrong, None if usable."""
    missing = {"doc_id", "kind", "domain", "para_start", "para_end", "prompt"} - u.keys()
    if missing:
        return f"missing {sorted(missing)}"
    if u["kind"] not in ("essay", "passage"):
        return f"bad kind {u['kind']}"
    if u["domain"] not in ("essay", "conversation", "explanation"):
        return f"bad domain {u['domain']}"
    if doc is None:
        return "unknown doc_id"
    n = len(paragraphs(doc["text"]))
    if not 0 <= u["para_start"] <= u["para_end"] < n:
        return f"span {u['para_start']}-{u['para_end']} out of range 0-{n - 1}"
    words = len(span_text(doc, u["para_start"], u["para_end"]).split())
    if u["kind"] == "passage" and not 80 <= words <= 900:
        return f"passage has {words} words"
    if u["kind"] == "essay" and words < 300:
        return f"essay has {words} words"
    if len(u["prompt"].split()) < 3:
        return "prompt too short"
    return None


def prep(a):
    """One numbered-paragraph file per doc, and batch-<tag>-NNN.list files of ~N words."""
    done = {u["doc_id"] for u in load_units(a.units)}
    only = set(filter(None, a.only.split(",")))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    batches, cur, words = [], [], 0
    for d in map(json.loads, open(a.corpus)):
        if d["doc_id"] in done or only and d["source"] not in only:
            continue
        paras = paragraphs(d["text"])
        (out / f"{d['doc_id']}.txt").write_text(
            f"doc_id: {d['doc_id']}\nsource: {d['source']}\ntitle: {d['title']}\n"
            f"url: {d['url']}\nwords: {d['words']}\nparagraphs: {len(paras)}\n\n"
            + "\n\n".join(f"[{i}] {p}" for i, p in enumerate(paras)))
        if cur and words + d["words"] > a.batch_words:
            batches.append(cur)
            cur, words = [], 0
        cur.append(d["doc_id"])
        words += d["words"]
    batches += [cur] if cur else []
    for i, b in enumerate(batches):
        (out / f"batch-{a.tag}-{i:03d}.list").write_text(
            "".join(f"{out.resolve()}/{d}.txt\n" for d in b))
    print(json.dumps({"docs": sum(map(len, batches)), "batches": len(batches)}))


def manifest(a):
    """Write data/manifest.jsonl, data/holdout.json (5% of docs by hash, kept out of
    training) and the prompts of held-out essays for human-reference evals."""
    corpus = {d["doc_id"]: d for d in map(json.loads, open(a.corpus))}
    units = load_units(a.units)
    ok, bad = [], []
    for u in units:
        if u.get("skip"):
            continue
        err = validate_unit(u, corpus.get(u.get("doc_id")))
        if err:
            bad.append((u, err))
        else:
            ok.append(u)
    ids = sorted({u["doc_id"] for u in ok})
    holdout = [d for d in ids if int(hashlib.sha1(d.encode()).hexdigest(), 16) % 20 == 0]
    keys = ("doc_id", "source", "url", "title", "words", "sha256")
    Path("data/manifest.jsonl").write_text(
        "".join(json.dumps({k: corpus[d][k] for k in keys}, ensure_ascii=False) + "\n"
            for d in ids))
    Path("data/holdout.json").write_text(json.dumps(holdout, indent=0))
    Path("data/eval/human_ref_prompts.jsonl").write_text("".join(
        json.dumps({"id": f"href-{u['doc_id']}:{u['para_start']}-{u['para_end']}",
            "domain": u["domain"], "prompt": u["prompt"]}, ensure_ascii=False) + "\n"
        for u in ok if u["doc_id"] in holdout and u["kind"] == "essay"))
    skips = [u for u in units if u.get("skip")]
    words = Counter()
    for u in ok:
        doc = corpus[u["doc_id"]]
        words[u["kind"]] += len(span_text(doc, u["para_start"], u["para_end"]).split())
    print(json.dumps({"docs_annotated": len({u["doc_id"] for u in units}),
        "docs_skipped": len(skips), "units_ok": len(ok), "units_bad": len(bad),
        "by_kind": Counter(u["kind"] for u in ok),
        "by_domain": Counter(u["domain"] for u in ok),
        "by_source": Counter(corpus[u["doc_id"]]["source"] for u in ok),
        "words_by_kind": words, "holdout_docs": len(holdout),
        "skip_reasons_sample": [s.get("reason", "")[:80] for s in skips[:15]]}, indent=1))
    for u, e in bad[:40]:
        print("BAD", u.get("doc_id"), u.get("para_start"), u.get("para_end"), e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "manifest"])
    ap.add_argument("corpus")
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("--units", default="data/backtranslations")
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="w")
    ap.add_argument("--batch-words", type=int, default=20000)
    a = ap.parse_args()
    {"prep": prep, "manifest": manifest}[a.cmd](a)
