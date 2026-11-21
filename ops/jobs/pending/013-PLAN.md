# Plan 013: from "beats the prompt" to "beats human essays"

Status: plan only. Nothing here is approved to launch. Each Modal job and each spend
over $1 needs the owner's go-ahead. Written so a fresh model can execute it.

## Goal, fixed before anything runs
Metric: human essay vs fine-tune + its system prompt, on the 68 held-out essay prompts
(`data/eval/human_ref_prompts.jsonl`; texts in the session scratchpad `href_human.jsonl`),
both orders, DeepSeek V4.1 Flash. Report the human win rate and a paired comparison against
the previous model on the same prompts.
- Now: humans win 0.65 against pol2 + style prompt (probe 012).
- **Reach**: the 95% interval of the human win rate includes 0.50.
- **Beat**: the interval sits below 0.50 (about 0.42 or lower at n=68).
- DeepSeek is softer on humans than Claude (human vs base: DeepSeek 0.89, Opus 0.96). A
  DeepSeek "reach" is confirmed by Sonnet on all 68 prompts, both orders, and a final
  claim of "beat" by one Opus check. Without those, report it as DeepSeek-only.
- Leakage checked: none of the 68 source documents appear in any training prompt set or
  pair file; they are in `data/holdout.json`.

## What we know (all measured)
1. **Why humans still win** (DeepSeek's reasons on probe 012, 97 human-win verdicts vs 39
   model-win): substance, not formatting. Humans work through the hard case, test the idea
   on concrete cases, admit uncertainty ("I don't know what principle rules..."), and
   develop each claim. The model reads as "manifesto": slogan lines ("The office is not a
   perk. It is the operating system"), confident claims left undeveloped, thin coverage of
   what was asked, and factual slips (a misnamed Watergate figure, "Red Hat owns
   OpenStack"). Median length: human 1,546 words, model 644. The model wins when the human
   piece is padded, off-request or rambling.
2. **Imitating humans directly failed twice.** SFT on human essays (sft1, sft2) and DPO
   with the human essay as chosen (dpo1) all collapsed on essays: 0.12, 0.08 and 0.05 vs
   base (Opus), with long looping answers (1,500 to 1,800 words, up to 21 of 68
   truncated). Do not put human essays in as chosen responses again.
3. **Ranking the model's own samples works but slowly.** Humans: 0.89 (base) to 0.81
   (pol2), paired gain 0.085. The style prompt adds a further 0.16 on top. Selection can
   only pick among what the model already writes, and what is missing is ideas.
4. **The prompt pool is used up.** All 1,200 training prompts were used in rounds 0-2.
   Fresh: 1,127 essay prompts backtranslated from corpus essays
   (`data/backtranslations/`, kind essay, outside the holdout), plus 1,416 passage prompts
   (conversation, explanation). None overlap the old pool.
5. **Judges.** Jev agrees with Sonnet on only 72% of pairs between two prompted samples
   (89% when very confident, 12% of pairs). DeepSeek: 79% on 47 such pairs. True DeepSeek
   cost is about $0.0028 per comparison (1.6x the logged figure).

## Conjecture this plan tests
The missing quality is thinking, not style. If the model reasons before it writes
(Qwen's thinking mode) and is told what substance looks like, its essays get closer to
human. Selecting those essays and training the non-thinking model on the final answers
moves the thinking into the writing itself. This is falsifiable cheaply (step 1) before
any training.

## Step 1: probe 013, candidate sources (sampling only)
Code changes first (small, follow the style guide):
- `owl/generate.py`: a run option `thinking: true` sets `enable_thinking` and keeps only
  the text after `</think>` as the response (today it only strips the tags, which would
  keep the reasoning). Raise `max_tokens` for those runs to 8192 so thinking does not
  truncate the essay; record truncation.
- `configs/substance_system_prompt.txt`: the style prompt plus the diagnosis: work through
  the hardest objection, test claims on concrete cases, say where you are unsure, check
  facts you state, develop each point at the length the topic needs, no slogan lines.
  Keep it consistent with `owl/rubric.py`.

Job (H200, two model loads, about $1.20), all on the 68 essay prompts:
| Run | Model | Prompt | Thinking |
|---|---|---|---|
| S0 | base | style | no (missing reference: how much of 0.65 is the prompt) |
| S1 | pol2 | style | yes |
| S2 | pol2 | substance | no |
| S3 | pol2 | substance | yes |

Judge each against the human essays, both orders, DeepSeek: 4 x 136 comparisons, about
$1.50. Compare each paired against pol2 + style (probe 012, human win 0.65).

Predictions: S0 humans 0.72; S1 0.58; S2 0.60; S3 0.52.

Decision: the source with the lowest human win rate whose paired gain over pol2 + style
has a 95% interval above zero becomes round 3's candidate source. If none qualifies,
the conjecture is refuted; round 3 falls back to pol2 + style with essay-weighted prompts,
and the plan should be revisited before spending more.

## Step 2: round 3 candidates (sampling only, about $1.50)
- Prompts: 400 fresh backtranslated prompts (300 essay, 50 conversation, 50 explanation),
  drawn with a fixed seed, written to `data/prompts/train_r3.jsonl`. Exclude holdout docs.
- 4 samples per prompt from the winning source, temperature 0.8.

## Step 3: select pairs (about $2.50)
- Jev tournament over the 4 samples (`owl.judging prep tournament`, `owl.jev`): $0.30.
- For each prompt take Jev's best and worst; DeepSeek judges that pair in both orders;
  keep only pairs where both orders agree with Jev (about $2.20).
- Predicted survivors: 250 to 300. Below 200: report and stop before training.
- Check length: chosen should not be systematically shorter (the rubric's "longer is not
  better" must not turn into "shorter is better"; humans write 2.4x longer).

## Step 4: train and evaluate round 3 (one job, about $4.50)
- Code change: `owl/train.py:52` puts the run's system prompt into the DPO prompt, so the
  model is trained the way it is deployed. If thinking was used for candidates, train on
  the final answer only, without thinking.
- DPO pol2 -> pol3, same hyperparameters as round 2.
- Sample in the same job: pol3 + prompt on the 68 essay prompts (without thinking, and
  with it if the source used it), pol3 + prompt on the 180 eval prompts, capability set,
  and round 4 candidates (so a continuing loop needs one job per round).
- Judge (about $2.00): humans vs pol3 (68, both orders), pol3 + prompt vs pol2 + prompt
  (180, both orders), capability vs base; Sonnet check on 30 prompts.

Predictions: humans vs pol3 + prompt 0.55 to 0.60 (from 0.65); pol3 vs pol2, both
prompted, 0.58.

## Continue or stop
Each further round repeats steps 3 and 4 (about $7 a round: $4.50 Modal, $2.50 OpenRouter).
- Continue while the paired human gain per round has an interval above zero, capability
  does not drop, and the new model beats the previous one with prompts.
- Stop after round 6 in any case, or when a round gives no measurable gain.
- At "reach" by DeepSeek: Sonnet on all 68 prompts, both orders. At "beat": one Opus
  check, retry any blocked chunk on Sonnet.

## Honest expectations
- One round will not reach 0.50. A good source may get round 3 to about 0.55; after that
  rounds have given about 0.03 to 0.08 each, so reaching 0.50 by DeepSeek is plausible in
  2 to 4 rounds. Beating top human bloggers by Opus's standard is unlikely on this budget,
  and the plan says so rather than claiming a judge-specific win.
- Risk: a judge-specific win. Thinking-mode essays may please DeepSeek in ways Claude does
  not share; the Sonnet checks are there to catch that.

## Budget
| Item | Cost |
|---|---|
| Step 1 probe | $1.20 Modal + $1.50 OpenRouter |
| Steps 2-4, round 3 | $6.00 Modal + $4.50 OpenRouter |
| Each later round | $4.50 Modal + $2.50 OpenRouter |
| Three rounds in total | about $20 Modal, $12 OpenRouter |
The OpenRouter key has $3.08 left: enough for step 1 only. Raise the cap by about $10
before step 3.

## Rules carried over
- Launch by copying a spec into `ops/jobs/` (a `git mv` rename runs nothing).
- At most one relaunch per failed job, after the cause is understood. Cancel a job stuck
  past 45 minutes.
- Write predictions before each run; report results against them before interpreting.
