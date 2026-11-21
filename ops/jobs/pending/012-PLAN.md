# Plan 012: probe Round 2 before any round 3

Written for whoever executes it (a fresh model can follow it without the earlier
conversation). The owner has approved launching this probe. Nothing else is approved:
do not launch round 3 (`011-r3.yaml`) or any other Modal job.

## Why
Round 2 (pol2) beats base (0.62, Sonnet) but loses to base + style prompt (0.36,
DeepSeek). 311 of 357 round-2 training pairs are "style-prompt sample beats plain
sample", so the loop is mostly distilling the prompt. Before training again we need a
baseline and a stop signal.

## What is measured, and predictions written before the run
| Test | Samples | Judge | Prediction |
|---|---|---|---|
| A. pol2 + style prompt vs base + style prompt, 180 prompts, both orders | new `eval_pol2_styled.jsonl` vs `ops/results/001-baseline/eval_base_styled.jsonl` | DeepSeek | about 0.50 |
| B. human essays vs pol2, 68 prompts, both orders | human file (below) vs new `href_pol2.jsonl` | DeepSeek | humans win about 0.85 |
| C. human essays vs pol2 + style prompt, same | human file vs new `href_pol2_styled.jsonl` | DeepSeek | humans win about 0.80 |

Reference points: DeepSeek gives human vs plain base 0.893. Test A at 180 prompts has a
margin of error of about +-0.07 (two standard errors), tests B and C about +-0.08.

Reading the results:
- A >= 0.57: the fine-tune adds to the prompt. A <= 0.43: it hurts the prompted model.
  In between: indistinguishable from even.
- B or C <= 0.78: training has measurably closed the gap to human essays. Otherwise no
  detectable change; B and C are a yardstick, not a sharp test.

## Steps
1. Launch (about $1.50, about 20 minutes, H200, sampling only):
   ```bash
   git mv ops/jobs/pending/012-r2-probe.yaml ops/jobs/012-r2-probe.yaml
   git commit -m "Launch 012-r2-probe" && git push -u origin claude/bold-lovelace-nkhae0
   ```
   The push starts the GitHub Actions workflow `modal-job`. If the permission check
   blocks the push, stop and tell the owner; do not work around it.
2. Watch the run with the GitHub MCP tools (`actions_list`, `get_job_logs`). Results land
   in `ops/results/012-r2-probe/` on the branch; `git pull` to get them. Expect 180, 68
   and 68 rows.
3. Judge with DeepSeek (about $1.30 total; the key has about $4.80). `S` is the session
   scratchpad. The human essays are at `$S/href_human.jsonl` (68 rows, not committed). If
   that file is gone, skip B and C and say so; do not rebuild it by guesswork.
   ```bash
   PY=/tmp/claude-0/venv/bin/python; export PYTHONPATH=.
   R=ops/results/012-r2-probe; M=deepseek/deepseek-v4.1-flash-1017
   $PY -m owl.judging prep vs --a ops/results/001-baseline/eval_base_styled.jsonl \
       --b $R/eval_pol2_styled.jsonl --out $S/p012/a
   $PY -m owl.judging prep vs --a $S/href_human.jsonl --b $R/href_pol2.jsonl --out $S/p012/b
   $PY -m owl.judging prep vs --a $S/href_human.jsonl --b $R/href_pol2_styled.jsonl \
       --out $S/p012/c
   for d in a b c; do
       $PY -m owl.jev --dir $S/p012/$d --model $M --out ds --workers 8
       $PY -m owl.judging collect vs --dir $S/p012/$d
   done
   ```
   `owl.jev` resumes: rerun the same command until `judged` is 0. `collect vs` reports the
   challenger's win rate; in B and C the human win rate is 1 minus that.
4. Record: write `ops/results/diag/probe_012.json` with the three win rates, by-domain
   numbers, order consistency, cost, and the predictions above next to the results. Copy
   the verdict files beside it as `verdicts_deepseek_012_{a,b,c}.jsonl`. Commit and push.
5. Report to the owner in plain language: each result against its prediction, what it
   rules out, and the recommended round 3 (below). Then stop and wait.

## Failure rules (budget is tight; reliability matters more than speed)
- If the Modal job fails, read `modal.log` and the job logs and report the cause. Relaunch
  at most once, and only after the cause is understood and fixed.
- If the workflow is still running after 45 minutes, look at the logs; cancel if it is
  stuck rather than letting it bill.
- If DeepSeek returns no verdict for more than 5% of comparisons after three reruns,
  report the partial result with its count instead of spending more.
- Hard caps for this plan: $3.50 Modal (one launch plus one relaunch), $2.50 OpenRouter.

## What round 3 would be (for the report only; needs a new go-ahead)
Both outcomes of test A point the same way: the only signal beyond the prompt is in
comparing style-prompt samples with each other. Facts already measured
(`ops/results/diag/jev_styled_calibration.json`):
- Round 2 yields only 25 such pairs, so new sampling is needed (for example 4 style-prompt
  samples per prompt, best against worst).
- Jev is a weak judge of these pairs: 72% agreement with Sonnet (89% on base vs styled),
  and 89% only on the 12% of pairs where it is very confident. DeepSeek: 79% on 47 pairs.
  The design must filter for confidence or use both orders, and say how many pairs survive.
- `owl/train.py` builds DPO prompts from the user turn only; training with the system
  prompt in context needs a small change there.
- If A <= 0.43, recommend stopping fine-tuning on this loop rather than a round 3.
