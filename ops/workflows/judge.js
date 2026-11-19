export const meta = {
  name: 'owl-judge',
  description: 'Pairwise-judge response chunks under the writing rubric, retry short chunks',
  phases: [{ title: 'Judge' }, { title: 'Retry' }],
}
// args: {dir, chunks: [{name, n}], model?: 'sonnet' | 'opus'} (see owl/judging.py write_chunks)
// Sonnet for training-round tournaments, Opus for held-out evals and calibration.
const { dir, chunks, model } = args
const opts = model ? { model } : {}
const SCHEMA = {
    type: 'object', properties: { written: { type: 'integer' } }, required: ['written'] }

function prompt(c, retry) {
  const src = `${dir}/chunks/${c.name}.txt`
  const out = `${dir}/verdicts/${c.name}${retry ? '-retry' : ''}.jsonl`
  return `You are a demanding, fair judge of writing quality. Read ${src}. It starts with a RUBRIC, then contains ${c.n} COMPARISONs, each with a request and two responses (A and B).

For EVERY comparison, apply the rubric and decide which response is better written in the full sense (sounds right AND is right). Judge each comparison on its own merits; they are unrelated. Do not let position, length, or formatting sway you by themselves. Think about each one properly: read both responses in full before deciding.
${retry ? `\nSome comparisons in this chunk were missed earlier. Look at ${dir}/verdicts/${c.name}.jsonl if it exists and only judge the cids that are not already there.\n` : ''}
Write one JSON object per comparison to ${out}, one per line:
{"cid": "<cid exactly as given>", "reason": "<2-4 sentences naming the decisive differences>", "verdict": "A>>B" | "A>B" | "A=B" | "B>A" | "B>>A"}

Use ">>" only when the gap would be obvious to any careful reader. Use "A=B" sparingly, only when you truly cannot prefer one. Write the file incrementally if that helps (append as you go). When done, validate it: python3 -c "import json;[json.loads(l) for l in open('${out}')]" and fix any malformed lines.

Return {written: <number of verdict lines you wrote>}.`
}

phase('Judge')
const first = await parallel(chunks.map(c => () =>
    agent(prompt(c, false), { label: c.name, phase: 'Judge', schema: SCHEMA, ...opts })))
const short = chunks.filter((c, i) => !first[i] || first[i].written < c.n)
if (short.length) {
  log(`${short.length} chunk(s) short or failed; retrying`)
  phase('Retry')
  await parallel(short.map(c => () => agent(prompt(c, true),
      { label: `${c.name} retry`, phase: 'Retry', schema: SCHEMA, ...opts })))
}
const written = first.filter(Boolean).reduce((s, r) => s + r.written, 0)
return { chunks: chunks.length, retried: short.map(c => c.name), written }
