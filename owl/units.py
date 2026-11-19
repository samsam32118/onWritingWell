"""Turn scraped documents into training units.

A *unit* is (prompt, response) where the response is a human-written span of a document.
Spans are stored as paragraph index ranges so the repo can commit the prompts and spans
without committing the copyrighted text:

    data/backtranslations/*.jsonl   one row per unit, written by the annotator agents
        {doc_id, kind: "essay"|"passage", domain, para_start, para_end, prompt}
        or {doc_id, skip: true, reason}

On Modal, `materialize` joins these with the corpus to produce chat-format examples.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

NOTES_HEADING = re.compile(r"^(notes|footnotes|references|thanks|acknowledg(e)?ments)\s*:?$", re.I)


def paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]


def numbered(doc: dict) -> str:
    """Render a document with [i] paragraph markers for the annotator agents."""
    paras = paragraphs(doc["text"])
    head = f"doc_id: {doc['doc_id']}\nsource: {doc['source']}\ntitle: {doc['title']}\nurl: {doc['url']}\nwords: {doc['words']}\nparagraphs: {len(paras)}\n\n"
    return head + "\n\n".join(f"[{i}] {p}" for i, p in enumerate(paras))


def span_text(doc: dict, start: int, end: int) -> str:
    """Paragraphs start..end inclusive, stopping early at a Notes/Thanks heading."""
    paras = paragraphs(doc["text"])
    out = []
    for p in paras[start : end + 1]:
        if NOTES_HEADING.match(p.strip()):
            break
        out.append(p)
    return "\n\n".join(out).strip()


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_units(directory: str | Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(Path(directory).glob("*.jsonl")):
        rows += load_jsonl(p)
    return rows


def validate_unit(u: dict, doc: dict | None) -> str | None:
    """Return an error string, or None if the unit is usable."""
    if u.get("skip"):
        return None
    for k in ("doc_id", "kind", "domain", "para_start", "para_end", "prompt"):
        if k not in u:
            return f"missing {k}"
    if u["kind"] not in ("essay", "passage"):
        return f"bad kind {u['kind']}"
    if u["domain"] not in ("essay", "conversation", "explanation"):
        return f"bad domain {u['domain']}"
    if doc is None:
        return "unknown doc_id"
    n = len(paragraphs(doc["text"]))
    if not (0 <= u["para_start"] <= u["para_end"] < n):
        return f"span {u['para_start']}-{u['para_end']} out of range 0-{n - 1}"
    words = len(span_text(doc, u["para_start"], u["para_end"]).split())
    if u["kind"] == "passage" and not (80 <= words <= 900):
        return f"passage has {words} words"
    if u["kind"] == "essay" and words < 300:
        return f"essay has {words} words"
    if len(u["prompt"].split()) < 3:
        return "prompt too short"
    return None


def materialize(units: list[dict], corpus: dict[str, dict], max_words: int = 4500) -> tuple[list[dict], list[str]]:
    """Return (examples, errors). Example: {id, domain, kind, source, messages}."""
    examples, errors = [], []
    for u in units:
        if u.get("skip"):
            continue
        doc = corpus.get(u["doc_id"])
        err = validate_unit(u, doc)
        if err:
            errors.append(f"{u.get('doc_id')}:{u.get('para_start')}: {err}")
            continue
        text = span_text(doc, u["para_start"], u["para_end"])
        if len(text.split()) > max_words:
            errors.append(f"{u['doc_id']}: span too long ({len(text.split())} words)")
            continue
        examples.append(
            {
                "id": f"{u['doc_id']}:{u['para_start']}-{u['para_end']}",
                "doc_id": u["doc_id"],
                "domain": u["domain"],
                "kind": u["kind"],
                "source": doc["source"],
                "messages": [
                    {"role": "user", "content": u["prompt"].strip()},
                    {"role": "assistant", "content": text},
                ],
            }
        )
    return examples, errors
