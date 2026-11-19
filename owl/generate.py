"""Sample responses with vLLM, thinking disabled, same chat template as training.

Input rows need {id, prompt}; other fields are carried through. Output rows add
{model_tag, sample_idx, response, finish_reason}.

    python -m owl.generate --model PATH --prompts p.jsonl --out o.jsonl --n 4 --tag sft1
"""
import argparse
import json

from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--max-model-len", type=int, default=12288)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--system", default=None, help="file with a system prompt (prompting baseline)")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.prompts) if l.strip()][:a.limit]
# Text only: zero multimodal items skips profiling the vision encoder.
llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len,
    gpu_memory_utilization=0.90, limit_mm_per_prompt={"image": 0, "video": 0},
    enable_prefix_caching=True)
sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, top_k=20,
    max_tokens=a.max_tokens, seed=0)
system = [{"role": "system", "content": open(a.system).read()}] if a.system else []
outs = llm.chat([system + [{"role": "user", "content": r["prompt"]}] for r in rows], sp,
    chat_template_kwargs={"enable_thinking": False}, use_tqdm=True)
with open(a.out, "w") as f:
    for r, o in zip(rows, outs):
        for i, c in enumerate(o.outputs):
            text = c.text.replace("<think>", "").replace("</think>", "").strip()
            f.write(json.dumps({**r, "model_tag": a.tag, "sample_idx": i, "response": text,
                "finish_reason": c.finish_reason}, ensure_ascii=False) + "\n")
