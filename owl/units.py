"""Training units: (prompt, human-written span of a document).

Spans are paragraph index ranges, so the repo can commit prompts and spans without the
copyrighted text. data/backtranslations/*.jsonl, written by the annotator agents, holds
    {doc_id, kind: essay|passage, domain, para_start, para_end, prompt}
    or {doc_id, skip: true, reason}
Modal joins them with the corpus in modal_app.build_sft.

    python -m owl.units prep CORPUS OUTDIR [--only src,...]   numbered docs + batch lists
    python -m owl.units manifest CORPUS                      validate units, write manifest
    python -m owl.units remap OLD_CORPUS --new NEW_CORPUS    move spans to re-extracted text
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
        # Double spaces after full stops are a typewriter habit; leave code alone.
        out.append(p if p.startswith("```") else re.sub(r"(?<=\S) {2,}", " ", p))
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


def remap(a):
    """Rewrite unit spans in place after re-extraction changed paragraph splitting.

    Paragraphs are aligned on their alphanumeric characters (formatting and spacing may
    differ). A span keeps the new paragraphs that cover it; if covering them adds more than
    15% text the unit is dropped, since its boundary fell inside a real paragraph."""
    norm = lambda t: re.sub(r"[^0-9a-z]", "", t.lower())
    old = {d["doc_id"]: d for d in map(json.loads, open(a.corpus))}
    new = {d["doc_id"]: d for d in map(json.loads, open(a.new))}
    spans = {}
    for doc_id in old.keys() & new.keys():
        stream = "".join(map(norm, paragraphs(new[doc_id]["text"])))
        nstart = [0]
        for p in paragraphs(new[doc_id]["text"]):
            nstart.append(nstart[-1] + len(norm(p)))
        # A paragraph missing from the new text (e.g. a "Thanks to..." line now dropped as
        # furniture) only invalidates spans that start or end on it.
        ostart, cur = [], 0
        for p in paragraphs(old[doc_id]["text"]):
            at = stream.find(norm(p), cur) if norm(p) else -1
            ostart.append((at, at + len(norm(p))) if at >= 0 else None)
            cur = at + len(norm(p)) if at >= 0 else cur
        spans[doc_id] = (ostart, nstart)
    stats = Counter()
    for f in sorted(Path(a.units).glob("*.jsonl")):
        out = []
        for u in map(json.loads, open(f)):
            if u.get("skip"):
                out.append(u)
                continue
            if u["doc_id"] not in old:
                out.append(u)  # not part of this re-extraction
                continue
            ostart, nstart = spans.get(u["doc_id"], ([], []))
            if u["para_end"] >= len(ostart) or not ostart[u["para_start"]] \
                    or not ostart[u["para_end"]]:
                stats["unmappable"] += 1
                continue
            lo, hi = ostart[u["para_start"]][0], ostart[u["para_end"]][1]
            s = max(i for i in range(len(nstart) - 1) if nstart[i] <= lo)
            e = min(i for i in range(len(nstart) - 1) if nstart[i + 1] >= hi)
            extra = (nstart[e + 1] - nstart[s]) - (hi - lo)
            if extra > 0.15 * max(hi - lo, 1):
                stats["dropped_boundary"] += 1
                continue
            stats["clean" if extra == 0 else "widened"] += 1
            out.append({**u, "para_start": s, "para_end": e})
        f.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in out))
    print(json.dumps(stats))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "manifest", "remap"])
    ap.add_argument("corpus")
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("--units", default="data/backtranslations")
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="w")
    ap.add_argument("--batch-words", type=int, default=20000)
    ap.add_argument("--new")
    a = ap.parse_args()
    {"prep": prep, "manifest": manifest, "remap": remap}[a.cmd](a)
