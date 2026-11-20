export const meta = {
  name: 'owl-retention',
  description: 'Write retention examples (prompt + correct, plainly written answer) by kind',
  phases: [{ title: 'Write' }],
}
// args: {outDir, evalPath, perAgent, jobs: [{kind, part, model, brief}]}
// Retention examples keep SFT on long essays from costing correctness, brevity and format
// obedience. Answers model the target voice: right first, plain, and they stop when done.
const { outDir, evalPath, perAgent, jobs } = args
const SCHEMA = {
    type: 'object', properties: { written: { type: 'integer' } }, required: ['written'] }

function prompt(j) {
  const out = `${outDir}/${j.kind}-${j.part}.jsonl`
  return `You are writing training data for a language model that is being fine-tuned to write well: like a thoughtful, careful person, not a template-driven assistant. This batch keeps it CORRECT and BRIEF on tasks where essays are the wrong shape.

Kind: ${j.kind}. ${j.brief}

Write ${perAgent} examples. Make them diverse in topic, difficulty and phrasing (some terse, some chatty, some precise). This is part ${j.part}; use your own varied choices so the two parts of this kind do not repeat each other. Do NOT reuse or closely paraphrase any prompt in ${evalPath} (read it first; it is a held-out test set).

How every answer must be written:
- Correct above all. Check every number, fact and line of code before writing it down. For math and logic, work the problem fully yourself first, then double-check by another route.
- Lead with the answer or the work that gets there; no preamble, no restating the question, no "Great question".
- Plain prose. No headers, no bold, no bullet lists unless the request asks for a list. Code goes in a fenced block tagged with its language.
- Show reasoning when it is needed to trust the answer (math, logic), compactly, in the order a careful person would think it. Never "Wait" / "Let me re-evaluate" in the final answer.
- Obey format instructions exactly ("nothing else" means nothing else: no code fence, no trailing remark).
- Stop when done. No summary, no offer of further help.

Write JSON lines to ${out}, one object per line:
{"id": "ret-${j.kind}-${j.part}-NNN", "kind": "${j.kind}", "prompt": "...", "response": "..."}
(NNN = 001, 002, ...). Write incrementally if that helps. When done, validate: python3 -c "import json;r=[json.loads(l) for l in open('${out}')];assert all(x['prompt'] and x['response'] for x in r);print(len(r))" and fix any problems.

Return {written: <number of lines>}.`
}

phase('Write')
const res = await parallel(jobs.map(j => () =>
    agent(prompt(j), { label: `${j.kind}-${j.part}`, phase: 'Write', schema: SCHEMA,
        model: j.model })))
return { written: res.filter(Boolean).reduce((s, r) => s + r.written, 0),
    failed: jobs.filter((j, i) => !res[i]).map(j => `${j.kind}-${j.part}`) }
