# On Writing Well

Fine-tune **Qwen3.8-27B** to write well: essays that develop ideas, conversational replies,
and explanations that sound like a thoughtful person rather than an assistant.

"Writing well" here means what Paul Graham argues in his essay on good writing, which
motivates this project: good writing *sounds* right and *is* right, and the two are one
rope, not two rods. Rhythm that matches the ideas; the right true things, developed to the
right depth; nothing that catches when you reread it. That definition lives in
[`owl/rubric.py`](owl/rubric.py) and every judgment in the project uses it.

## Why this is hard, and the approach

Post-training for code and math rewards verifiable correctness, and the cheapest way to be
reliably correct is to be generic: hedge, enumerate, format, summarise. Voice flattens. So
we need (a) examples of the target voice, and (b) a reward that can tell "sounds right and
is right" apart from "looks thorough".

| Stage | What | Signal |
|---|---|---|
| 0. Corpus | Essays by writers who think on the page (Graham, Luu, Alexander, McKenzie, Evans, Olah, Karlsson, Spolsky, Sivers, …) — [`configs/sources.yaml`](configs/sources.yaml) | human text |
| 1. Backtranslation | Claude subagents read each essay and write the request it answers, plus self-contained passages that answer chat questions. Stored as paragraph spans. | human text as target |
| 2. SFT (LoRA) | Train on (request → human text). Thinking off; same chat template as inference. | imitation |
| 3. Offline DPO | chosen = human text, rejected = the base model's answer to the same request | contrast with the default voice |
| 4. Online rounds | Sample 4 answers per prompt from the current policy → Claude subagents judge round-robin under the rubric → DPO on (best, worst) with a margin → repeat | on-policy preference |
| Eval | Held-out prompts by topic; pairwise vs base (both orders); vs held-out human essays; slop tripwires; the owner's own labels | |

Guards against reward hacking: pairwise judging with randomised and swapped order,
explicit "longer is not better" in the rubric, length/format/stock-phrase tripwires
([`owl/slop.py`](owl/slop.py)) reported next to every win rate, an SFT (RPO) term in DPO,
and periodic calibration of the judge against human labels.

## Where things run

```
this repo (Claude Code)                        Modal (H200)
  scrape, annotate, judge (subagents)            download model, build corpus, SFT/DPO, vLLM sampling
  ops/jobs/<id>.yaml  ── push ──► GitHub Actions ──► modal run modal_app.py --spec ...
  ops/results/<id>/   ◄── commit ── Actions  ◄──── /vol/results/<id>/
```

* The human-written corpus is fetched on Modal from [`data/manifest.jsonl`](data/manifest.jsonl)
  (URL + text hash) and never committed: its authors hold the copyright. The repo holds only
  URLs, hashes, paragraph spans, and prompts we wrote. Everything else here is CC-BY-4.0.
* The essays are fetched once into the Modal volume (`/vol/corpus.jsonl`) and reused by every
  job. To remove them: `modal volume rm owl-vol /corpus.jsonl` (and `/data/` for the
  training sets built from them).
* Judging is done by Claude subagents in Claude Code, not by API calls.

## Layout

```
configs/sources.yaml        which writers, how to find their posts
owl/scrape.py               fetch + extract + clean
owl/units.py                paragraph spans <-> training examples
owl/rubric.py               the definition of writing well (judge prompt)
owl/slop.py                 judge-free tripwires (stock phrases, bullets, burstiness, length)
owl/judging.py              pairwise comparisons -> chunks for judge agents -> win rates / DPO pairs
owl/train.py                LoRA SFT, DPO, merge (TRL + PEFT)
owl/generate.py             vLLM sampling
modal_app.py                Modal functions + job runner
ops/jobs/*.yaml             job specs (pushing one runs it)
ops/results/<id>/           what came back
data/prompts/               eval + training prompt pools
data/backtranslations/      annotator output (spans + prompts, no text)
```

## Running a job

1. Add repository secrets `TOKEN_ID` and `TOKEN_SECRET` (a Modal token pair).
2. Commit a spec under `ops/jobs/`, e.g.

   ```yaml
   id: 010-sft1
   steps:
     - build_corpus: {}
     - build_sft: {holdout: data/holdout.json, out: sft1.jsonl}
     - train: {stage: sft, model: base, data: vol:data/sft1.jsonl, out_model: sft1}
     - generate: {model: sft1, prompts: data/prompts/eval.jsonl, tag: sft1, out: eval_sft1.jsonl}
   ```
3. Push to a `claude/**` branch. Results appear in `ops/results/010-sft1/`.

A list inside `steps` runs those steps in parallel.

## Local checks

```bash
pip install -r requirements-dev.txt
pytest -q
```
