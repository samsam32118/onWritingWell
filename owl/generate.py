"""Sample responses with vLLM, thinking disabled, same chat template as training.

One process loads the model once and runs every prompt set in --runs, a JSON list of
    {prompts, out, tag, n?, temperature?, max_tokens?, system?, limit?}
Input rows need {id, prompt}; other fields are carried through. Output rows add
{model_tag, sample_idx, response, finish_reason}.

    python -m owl.generate --model PATH --runs runs.json
"""
import argparse
import json

from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--runs", required=True)
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--max-model-len", type=int, default=12288)
ap.add_argument("--bf16", action="store_true", help="skip on-the-fly FP8 weight quantization")
a = ap.parse_args()

# FP8 weights roughly halve memory traffic, the bottleneck when sampling; every model is
# sampled the same way so comparisons stay fair. Text only: zero multimodal items skips
# profiling the vision encoder.
llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len,
    quantization=None if a.bf16 else "fp8", gpu_memory_utilization=0.90,
    limit_mm_per_prompt={"image": 0, "video": 0}, enable_prefix_caching=True)
for run in json.load(open(a.runs)):
    rows = [json.loads(l) for l in open(run["prompts"]) if l.strip()][:run.get("limit")]
    sp = SamplingParams(n=run.get("n", 1), temperature=run.get("temperature", 0.8), top_p=0.95,
        top_k=20, max_tokens=run.get("max_tokens", 4096), seed=0)
    system = [{"role": "system", "content": open(run["system"]).read()}] \
        if run.get("system") else []
    outs = llm.chat([system + [{"role": "user", "content": r["prompt"]}] for r in rows], sp,
        chat_template_kwargs={"enable_thinking": False}, use_tqdm=True)
    with open(run["out"], "w") as f:
        for r, o in zip(rows, outs):
            for i, c in enumerate(o.outputs):
                text = c.text.replace("<think>", "").replace("</think>", "").strip()
                f.write(json.dumps({**r, "model_tag": run["tag"], "sample_idx": i,
                    "response": text, "finish_reason": c.finish_reason},
                    ensure_ascii=False) + "\n")
