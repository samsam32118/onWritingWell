"""Sample responses with vLLM (thinking disabled, same template as training).

Input rows need {id, prompt} (extra fields are carried through). Output rows:
    {id, prompt, model_tag, sample_idx, response, finish_reason, ...extra}

    python -m owl.generate --model PATH --prompts p.jsonl --out o.jsonl --n 4 --tag sft1
"""
from __future__ import annotations

import argparse
import json


def main(argv=None):
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
    a = ap.parse_args(argv)

    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(a.prompts) if l.strip()][: a.limit]
    llm = LLM(
        model=a.model,
        tensor_parallel_size=a.tp,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=0.90,
        limit_mm_per_prompt={"image": 0, "video": 0},  # text only: skip the vision encoder profile
        enable_prefix_caching=True,
    )
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, top_k=20, max_tokens=a.max_tokens, seed=0)
    convs = [[{"role": "user", "content": r["prompt"]}] for r in rows]
    outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=True)
    with open(a.out, "w") as f:
        for r, o in zip(rows, outs):
            for i, c in enumerate(o.outputs):
                text = c.text.replace("<think>", "").replace("</think>", "").strip()
                f.write(json.dumps({**r, "model_tag": a.tag, "sample_idx": i, "response": text, "finish_reason": c.finish_reason}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
