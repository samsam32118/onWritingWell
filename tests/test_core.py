import json

from owl import judging, slop, units
from owl.scrape import clean_text


DOC = {
    "doc_id": "d1", "source": "s", "url": "u", "title": "t", "words": 0,
    "text": "March 2020\n\nFirst paragraph here [1] with a note.\n\nSecond one.\n\nNotes\n\n[1] The note.",
}


def test_clean_text_strips_furniture():
    raw = "Body text <sup>\\[return\\]</sup>\n\\[EDIT\\] more\nThanks\nto A and B for reading drafts of this.\nSpanish Translation"
    out = clean_text(raw)
    assert "return" not in out and "Translation" not in out and "reading drafts" not in out
    assert "[EDIT] more" in out


def test_span_text_stops_at_notes_and_drops_markers():
    t = units.span_text(DOC, 1, 4)
    assert t == "First paragraph here with a note.\n\nSecond one."


def test_validate_unit_ranges():
    u = {"doc_id": "d1", "kind": "passage", "domain": "conversation", "para_start": 1, "para_end": 9, "prompt": "why is that so"}
    assert "out of range" in units.validate_unit(u, DOC)


def test_slop_metrics_flag_stock_phrases():
    bad = slop.metrics("Great question! It's important to note that we must delve into the rich tapestry of this.")
    good = slop.metrics("The bus was late again. I walked.")
    assert bad["stock_per_1k"] > good["stock_per_1k"] == 0


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_vs_judging_cancels_position(tmp_path):
    a = [{"id": f"essay-{i}", "prompt": "p", "model_tag": "base", "sample_idx": 0, "response": "x"} for i in range(3)]
    b = [{**r, "model_tag": "new", "response": "y"} for r in a]
    _write(tmp_path / "a.jsonl", a)
    _write(tmp_path / "b.jsonl", b)
    comps = judging.prep_vs(str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), str(tmp_path / "j"))
    judging.write_chunks(comps, str(tmp_path / "j"), chunk=2)
    # A judge that always prefers position A is pure position bias: the result must be a tie.
    _write(tmp_path / "j" / "verdicts" / "v.jsonl", [{"cid": c["cid"], "verdict": "A>B"} for c in comps])
    res = judging.collect_vs(str(tmp_path / "j"))
    assert res["win_rate"] == 0.5 and res["order_consistency"] == 0.0


def test_tournament_pairs(tmp_path):
    rows = [{"id": "c-0", "prompt": "q", "model_tag": "p", "sample_idx": k, "response": f"r{k}", "finish_reason": "stop"} for k in range(4)]
    _write(tmp_path / "s.jsonl", rows)
    comps = judging.prep_tournament(str(tmp_path / "s.jsonl"), k=4)
    assert len(comps) == 6
    judging.write_chunks(comps, str(tmp_path / "t"))
    verdicts = [{"cid": c["cid"], "verdict": "A>>B" if c["a_key"] > c["b_key"] else "B>>A"} for c in comps]
    _write(tmp_path / "t" / "verdicts" / "v.jsonl", verdicts)
    res = judging.collect_tournament(str(tmp_path / "t"), str(tmp_path / "pairs.jsonl"))
    pair = json.loads(open(tmp_path / "pairs.jsonl").readline())
    assert res["pairs"] == 1 and pair["chosen"] == "r3" and pair["rejected"] == "r0"


def test_remap_merges_split_paragraphs(tmp_path):
    old = {"doc_id": "d", "text": "Intro.\n\nThis is\n\nless\n\nimportant.\n\nEnd here."}
    new = {"doc_id": "d", "text": "Intro.\n\nThis is *less* important.\n\nEnd here."}
    (tmp_path / "old.jsonl").write_text(json.dumps(old) + "\n")
    (tmp_path / "new.jsonl").write_text(json.dumps(new) + "\n")
    (tmp_path / "u").mkdir()
    units_rows = [{"doc_id": "d", "kind": "passage", "domain": "essay", "para_start": 1,
        "para_end": 4, "prompt": "p"}, {"doc_id": "d", "kind": "passage", "domain": "essay",
        "para_start": 2, "para_end": 2, "prompt": "q"}]
    (tmp_path / "u" / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in units_rows))
    import argparse
    units.remap(argparse.Namespace(corpus=str(tmp_path / "old.jsonl"), new=str(tmp_path / "new.jsonl"),
        units=str(tmp_path / "u")))
    rows = [json.loads(l) for l in open(tmp_path / "u" / "a.jsonl")]
    # the first span maps cleanly to new paragraphs 1-2; the one-word fragment span is dropped
    assert [(r["para_start"], r["para_end"]) for r in rows] == [(1, 2)]
