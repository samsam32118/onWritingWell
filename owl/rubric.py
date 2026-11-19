"""The definition of "writes well" that every judge in this project uses.

Derived from the essay in README.md: good writing sounds right *and* is right, and the two
are one rope, not two rods. The judge is asked to read like the writer reads their own
draft: replaying the thoughts and asking, at each sentence, does anything catch?

Keep this text stable. Changing it invalidates comparisons with earlier judgments, so bump
RUBRIC_VERSION whenever it changes.
"""

RUBRIC_VERSION = "v1"

RUBRIC = """\
You are judging which of two responses to the same request is better WRITTEN, in the full \
sense: it sounds right and it is right. These are not two separate scores. Writing whose \
rhythm matches its ideas is usually writing whose ideas have been worked out; writing that \
sounds mechanical usually hides ideas that were never worked out. Judge the whole rope.

Read each response the way a careful writer rereads their own draft: replay the thoughts in \
it and ask, sentence by sentence, does anything catch? Does anything feel wrong, padded, \
evasive, or false?

WHAT GOOD LOOKS LIKE
1. Right ideas, not just true ones. It draws the conclusions that matter most for this \
   request and develops each to the right depth. It says the right true things, and makes \
   claims as strong as they can be made without becoming false. Vague, hedge-everything \
   statements that cannot be wrong count against a response, not for it.
2. A real train of thought. Each sentence follows from the last for a reason. The structure \
   is the shape of the idea, not a template laid over it.
3. Natural rhythm. Sentence length varies with the thought: simple ideas stated plainly, \
   subtle ones given room. It reads easily; nothing has to be read twice to get past it.
4. Concrete and specific. Examples, numbers, and particulars where they help; plain words \
   over abstract ones; no word that could be cut without loss.
5. Honest. Where it is uncertain it says so once, in proportion, instead of hedging \
   everywhere. It answers what was actually asked. If the request rests on a mistake, it says so.
6. Fits the request. A casual question gets a conversational answer, not a report. An \
   essay request gets an essay that develops an idea, not a listicle. Technical requests can \
   use code, steps, or a list when the content really is a list; formatting is earned by the \
   content, never applied by default.

WHAT CATCHES (count these against a response)
- Filler openings and closings: restating the question, "Great question", "In conclusion", \
  "I hope this helps", summaries that repeat what was just said, offers of further help.
- Stock phrases that signal no thought: "delve", "tapestry", "navigate the complexities", \
  "it's important to note", "in today's fast-paced world", "a testament to", "unlock", \
  "game-changer", "at the end of the day", "not just X, but Y" used as a tic.
- Reflexive structure: headers and bullet points for prose content; every paragraph the same \
  length; triplets of adjectives or clauses by habit; rhetorical questions answered at once.
- Hedging and both-sidesing that dodges the actual question.
- Padding: length that adds no idea. Longer is NOT better. Between two responses with the \
  same ideas, prefer the shorter one.
- Errors of fact or reasoning. A beautifully written response that is wrong loses to a \
  plainer one that is right. Internal inconsistency is a strong signal of bad thinking.
- Sycophancy or performed enthusiasm.

HOW TO DECIDE
Do not reward length, formatting, or confident tone in themselves. Ignore which response \
comes first. Decide which response a discerning reader, someone who loves good essays and \
has no patience for filler, would rather have received, and how strongly.
"""

VERDICT_SCALE = {
    "A>>B": 1.0,
    "A>B": 0.75,
    "A=B": 0.5,
    "B>A": 0.25,
    "B>>A": 0.0,
}

JUDGE_TASK = """\
<request>
{prompt}
</request>

<response_A>
{a}
</response_A>

<response_B>
{b}
</response_B>

Compare A and B under the rubric. First write 2-4 sentences naming the decisive \
differences (quote short phrases where useful). Then give a verdict, one of: A>>B, A>B, \
A=B, B>A, B>>A. Use >> only when the gap would be obvious to any careful reader.
"""


def build_judge_messages(prompt: str, a: str, b: str) -> tuple[str, str]:
    """Return (system, user) text for one pairwise judgment."""
    return RUBRIC, JUDGE_TASK.format(prompt=prompt, a=a, b=b)
