"""Sample responses with vLLM, thinking disabled, same chat template as training.

One process loads the model once and runs every prompt set in --runs, a JSON list of
    {prompts, out, tag, n?, temperature?, max_tokens?, system?, limit?, thinking?,
     think_budget?, reasoning_effort?}
Input rows need {id, prompt}; other fields are carried through. Output rows add
{model_tag, sample_idx, response, finish_reason}. With thinking: true the model reasons
first; the reasoning goes in a `reasoning` field and only the text after </think> is the
response. No </think>, or nothing after it, is finish_reason "no_answer". With
think_budget: N (implies thinking) reasoning stops after N tokens, the think block is closed
for the model, and the answer gets max_tokens more; think_cut marks cut reasoning.

    python -m owl.generate --model PATH --runs runs.json
"""
import argparse
import json

from vllm import LLM, SamplingParams

# vLLM starts its engine with multiprocessing spawn, which re-imports this module: everything
# below must only run in the parent process.
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=12288)
    a = ap.parse_args()

    # Text only: zero multimodal items skips profiling the vision encoder.
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len,
        gpu_memory_utilization=0.90, limit_mm_per_prompt={"image": 0, "video": 0},
        enable_prefix_caching=True)
    for run in json.load(open(a.runs)):
        rows = [json.loads(l) for l in open(run["prompts"]) if l.strip()][:run.get("limit")]
        sp = SamplingParams(n=run.get("n", 1), temperature=run.get("temperature", 0.8), top_p=0.95,
            top_k=20, max_tokens=run.get("max_tokens", 4096), seed=0)
        system = [{"role": "system", "content": open(run["system"]).read()}] \
            if run.get("system") else []
        think = bool(run.get("thinking") or run.get("think_budget"))
        msgs = [system + [{"role": "user", "content": r["prompt"]}] for r in rows]
        # Qwen's template defaults thinking to "xhigh" effort ("think carefully ... consider
        # alternatives"), which drafts whole essays in the reasoning; medium adds no line.
        kw = {"enable_thinking": think}
        if think and run.get("reasoning_effort"):
            kw["reasoning_effort"] = run["reasoning_effort"]
        if run.get("think_budget"):
            # Hard budget: reason for at most think_budget tokens, then close the think block
            # ourselves and let the model write the answer in max_tokens more.
            tok = llm.get_tokenizer()
            heads = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
                for m in msgs]
            sp1 = SamplingParams(n=sp.n, temperature=sp.temperature, top_p=0.95, top_k=20,
                max_tokens=run["think_budget"], stop=["</think>"], seed=0)
            firsts = llm.generate(heads, sp1, use_tqdm=True)
            tails = [h + c.text + "\n</think>\n\n" for h, o in zip(heads, firsts)
                for c in o.outputs]
            sp2 = SamplingParams(n=1, temperature=sp.temperature, top_p=0.95, top_k=20,
                max_tokens=sp.max_tokens, seed=0)
            seconds = iter(llm.generate(tails, sp2, use_tqdm=True))
            for o in firsts:
                for i, c in enumerate(o.outputs):
                    c2 = next(seconds).outputs[0]
                    o.outputs[i] = type("Out", (), {"text": c.text + "</think>" + c2.text,
                        "finish_reason": c2.finish_reason, "cut": c.finish_reason == "length"})
            outs = firsts
        else:
            outs = llm.chat(msgs, sp, chat_template_kwargs=kw, use_tqdm=True)
        with open(run["out"], "w") as f:
            for r, o in zip(rows, outs):
                for i, c in enumerate(o.outputs):
                    reasoning, finish = "", c.finish_reason
                    text = c.text
                    if think:
                        reasoning, sep, text = text.partition("</think>")
                        reasoning = reasoning.replace("<think>", "").strip()
                        if not sep or not text.strip():
                            finish = "no_answer"
                    text = text.replace("<think>", "").replace("</think>", "").strip()
                    row = {**r, "model_tag": run["tag"], "sample_idx": i, "response": text,
                        "finish_reason": finish}
                    if think:
                        row["reasoning"] = reasoning
                    if run.get("think_budget"):
                        row["think_cut"] = c.cut
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
