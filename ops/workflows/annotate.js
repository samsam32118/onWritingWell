export const meta = {
  name: 'owl-annotate',
  description: 'Read essays and write the user requests they answer (instruction backtranslation), as paragraph spans',
  phases: [{ title: 'Annotate' }],
}
// args: {nBatches, annotDir, outDir, tag, model?} with batch lists from `python -m owl.units prep`
const { nBatches, annotDir, outDir, tag, model } = args
const opts = model ? { model } : {}
const batches = Array.from({ length: nBatches }, (_, i) => i)
const SCHEMA = { type: 'object', properties: { units: { type: 'integer' }, skipped: { type: 'integer' }, file: { type: 'string' }, notes: { type: 'string' } }, required: ['units', 'skipped', 'file'] }

const GUIDE = `You are preparing training data for a language model that should learn to write like the best human essayists: prose that sounds right and is right (natural rhythm, real trains of thought, concrete, honest, no filler). We have real essays by excellent writers. Your job is "instruction backtranslation": for each document, write the request a user might plausibly have sent for which this human text is an excellent reply, and pick which paragraphs form that reply.

Each document file shows numbered paragraphs: [0], [1], ... Produce UNITS:

1. ESSAY unit (at most one per document). Use it when the document is a coherent piece that develops an idea, 400-4500 words.
   - para_start / para_end: the first and last paragraph of the actual essay body. Exclude date lines, titles, "Update:" preambles, notes/footnotes sections, "Thanks to..." lines, translations, sign-up blurbs, comment sections.
   - domain: "essay" for argument/exploration; "explanation" when the piece is mainly teaching how something works.
   - prompt: what a curious person would type to get this piece. Give the topic and maybe an angle, but do NOT leak the essay's specific conclusions, structure, examples or memorable phrases (the model must learn to find the ideas itself). Vary the phrasing a lot across documents: "Write an essay on...", "Why do...? Write it up properly", "I want a blog post about...", a short paragraph of personal context ending in a request, etc. Occasionally specify a reader ("for someone about to start a PhD").
   - If the text is written in a strong personal context (e.g. "at YC we..."), the prompt can set that up ("You've funded hundreds of startups. Write an essay on..."), but keep it natural.

2. PASSAGE units (0-3 per document). A contiguous span of paragraphs, 120-600 words, that stands completely on its own as a reply to a question: no dangling references ("as I said above", "this", "the second reason" without the first, a name introduced earlier), no mid-argument start. Opening paragraphs often work; good mid-essay sections work if self-contained.
   - domain: "conversation" when it reads as a thoughtful personal reply (advice, opinion, a reaction); "explanation" when it explains a concept or mechanism.
   - prompt: a realistic chat message the passage answers well, often casual ("why do so many startups die from fundraising?", "ok but is it actually bad to procrastinate"). The passage should answer it directly; don't write a prompt the passage only half answers.
   - Quality over quantity: zero passages is fine.

SKIP a document (write {"doc_id": ..., "skip": true, "reason": "..."}) when it is mostly: announcements, link roundups, event notices, a list of quotes, tables/data dumps, code listings, transcripts or interviews, a book excerpt that depends on context, writing that leans on images/charts we can't see, or prose that is not good enough to imitate. Also skip if the writing is mostly out of date in a way that would teach false facts as current.

Output: JSON Lines, one object per unit:
{"doc_id": "<id>", "kind": "essay"|"passage", "domain": "essay"|"conversation"|"explanation", "para_start": <int>, "para_end": <int>, "prompt": "<request>"}
or the skip object above. Paragraph indices are the [n] numbers, inclusive.

Check your own work: for every unit, reread the first and last paragraph of the span and confirm it starts cleanly and ends cleanly. Then validate the file parses: python3 -c "import json;[json.loads(l) for l in open('FILE')]".`

const res = await parallel(batches.map((_, i) => () => {
  const file = `${outDir}/${tag}-${String(i).padStart(3, '0')}.jsonl`
  const files = `(the list of document files is in ${annotDir}/batch-${tag}-${String(i).padStart(3, '0')}.list, one path per line; read that list first)`
  return agent(`${GUIDE.replaceAll('FILE', file)}

Documents to annotate (read every one in full with the Read tool):
${files}

Write all units for these documents to ${file}. Return {units, skipped, file, notes} where notes is one or two sentences on anything odd (e.g. extraction problems you noticed).`, { label: `annotate ${tag}-${i}`, phase: 'Annotate', schema: SCHEMA, ...opts })
}))
const ok = res.filter(Boolean)
log(`${ok.length}/${batches.length} batches, ${ok.reduce((s, r) => s + r.units, 0)} units, ${ok.reduce((s, r) => s + r.skipped, 0)} skipped`)
return ok