# Plan 016: round 3, train the model to write the way it writes when it thinks

Status: plan only. Two Modal jobs, each needs the owner's go-ahead. Judging is Sonnet
(subagents, no OpenRouter spend). Written so a fresh model can execute it.

## Goal (unchanged)
Fine-tune Qwen3.8-27B to write well in Paul Graham's sense, in essays, conversation and
explanations, and reach then beat top human bloggers. Yardstick: human essays judged by
Sonnet (final claim by Opus); the 68 test prompts (`data/eval/human_ref_prompts.jsonl`) are
measured once per round, never used to choose anything; choices are made on the 68 dev
prompts (`data/eval/dev_prompts.jsonl`). Human texts rebuild from the corpus (013-PLAN.md).

## Where we are (Sonnet, 30 dev prompts, human win rate; lower is better)
| System | Humans win |
|---|---|
| pol2 + style prompt (P, the deployed setup) | 0.87 |
| pol2 + substance prompt + xhigh thinking (S3, finished samples) | 0.63 |
| pol2 + medium thinking, 2k budget (015) | 0.70 |
Head to head, Sonnet prefers S3 over P on 29 of 30 prompts (58 of 60 single verdicts).
The fine-tune matters with thinking on (S3 beats base + thinking). Medium effort loses
quality; xhigh's long reasoning does real work. Full numbers: `ops/results/diag/probe_01*.json`.

## Idea
Use pol2's best behaviour (xhigh thinking) as the teacher and its plain behaviour as the
contrast: DPO with chosen = the thinking essay, rejected = the plain style-prompt essay,
trained with the style prompt in context and no thinking. pol3 should then write closer
to its thinking essays without thinking, and pol3 with thinking should improve too.

## Free work already done
- `data/prompts/train_r3.jsonl`: 430 essay + 50 conversation + 50 explanation requests,
  fresh, holdout and dev documents excluded (seed 16).
- `owl/train.py`: a DPO pair may carry `system`; it is rendered in the prompt exactly as
  the non-thinking template does (checked against the real Qwen template).
- `owl/generate.py`: `think_budget` (hard reasoning cap, then the model writes the answer)
  and `reasoning_effort`. 59% of S3's samples finish reasoning under 8000 tokens.

## Steps
1. **Job 016a, candidates** (`ops/jobs/pending/016a-r3-candidates.yaml`), about $3.50.
   One pol2 load: thinking run (substance prompt, xhigh, `think_budget: 8000`,
   `max_tokens: 3500`) and plain run (style prompt), on the 530 prompts.
2. **Build pairs, free.** Keep a prompt when the thinking sample has `think_cut` false,
   `finish_reason` stop, a non-empty answer, no loop (no paragraph repeated), and the plain
   sample finished. Pair: `{id, prompt, system: <style prompt text>, chosen: think answer,
   rejected: plain answer}` -> `data/rounds/r3_pairs.jsonl`.
   - Essays: mechanical filter only (Sonnet preferred thinking 97% of the time).
   - Conversation and explanation: thinking is untested there, so Sonnet judges every pair
     in one order (about 60 comparisons, 4 subagents); keep only pairs where thinking wins.
   - Gate: stop and report if fewer than 250 pairs survive.
   - Length check: chosen will be longer (about 1,300 vs 650 words). Report the ratio; if
     chosen is more than 2.5x rejected in over a third of pairs, cap by dropping those.
3. **Job 016b, train and sample** (`ops/jobs/pending/016b-r3-train.yaml`), about $5.
   DPO pol2 -> pol3 (round-2 hyperparameters), then in one pol3 load: dev with style
   prompt, dev with xhigh thinking, test with style prompt, the 180 eval prompts with style
   prompt, capability set.
4. **Collapse guard before judging.** Stop if pol3 truncates over 5%, repeats paragraphs,
   or the training log shows saturating margins with chosen log-probability falling (the
   dpo1 failure). Report lengths against P.
5. **Judge with Sonnet** (subagents, both orders):
   - Dev 30: pol3 + style vs human, pol3 + thinking vs human, pol3 + style vs P.
   - 180 eval prompts (essay, conversation, explanation): pol3 + style vs pol2 + style,
     by domain, to catch regressions outside essays.
   - Capability set vs base.
   - Test 68: pol3 + style vs human, judged once and only if dev improves on P.
6. Record against the predictions, commit, report, stop.

## Predictions (written before any run)
- 016a: 55-65% of thinking samples usable; about 300 pairs.
- pol3 + style vs human (dev 30): 0.80 (P 0.87). pol3 + style beats P head to head 0.62.
- pol3 + xhigh thinking vs human: 0.58 (S3 0.63).
- No domain of the 180-prompt eval drops below 0.45 against pol2.
- One round is unlikely to bring the no-thinking model to S3's level (0.63).

## Stop rules
- Fewer than 250 pairs, collapse signs, or pol3 + style not beating P head to head: stop,
  keep pol2, report.
- Any domain below 0.45 against pol2 or capability falling: report before any next round.
- One relaunch at most per job, only after the cause is understood. Cancel a job stuck past
  its timeout.

## Costs
| Item | Cost |
|---|---|
| 016a candidates (about 45 minutes on H200) | about $3.50 Modal |
| 016b train + samples (about 60 minutes) | about $5 Modal |
| Judging | Sonnet subagents, $0 OpenRouter |
| Total | about $8.50 Modal |
Two go-aheads: 016a first, then 016b after the pair gate passes.

## Launch
Copy the spec into `ops/jobs/` (never `git mv`: a rename runs nothing), commit, push.
Confirm the GitHub Actions run finished (the Modal run exits with it) and report.
