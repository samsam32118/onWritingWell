"""LoRA SFT and DPO for Qwen3.8 (also works for any Qwen3.5-architecture model).

Both stages pre-render the chat template themselves with thinking disabled, so the tokens
we train on are exactly the tokens vLLM sees at inference:

    prompt     = <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
    completion = <response text><|im_end|>

LoRA is restricted to the language model's linear layers; the vision tower is untouched,
and `merge` writes a full checkpoint in the original format so vLLM can serve it.

    python -m owl.train sft   --model M --data sft.jsonl --out DIR [...]
    python -m owl.train dpo   --model M --data pairs.jsonl --out DIR [...]
    python -m owl.train merge --model M --adapter DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

LORA_TARGETS = (
    r".*language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))"
)


def render_prompt(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def render_completion(text: str) -> str:
    # Ends exactly at the EOS token so TRL does not append a second one.
    return text.strip() + "<|im_end|>"


def load_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _load_model(model_path: str, gradient_checkpointing: bool):
    import torch
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model


def _lora(r: int, alpha: int, dropout: float):
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")


def sft(a):
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = load_rows(a.data)
    random.Random(0).shuffle(rows)
    ds = Dataset.from_list(
        [{"prompt": render_prompt(tok, r["messages"][:-1]), "completion": render_completion(r["messages"][-1]["content"])} for r in rows]
    )
    cfg = SFTConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # transformers>=5: a float < 1 is a ratio
        max_length=a.max_length,
        completion_only_loss=True,
        bf16=_has_cuda(),
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        max_steps=a.max_steps,
    )
    trainer = SFTTrainer(
        model=_load_model(a.model, a.grad_ckpt),
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora(a.lora_r, a.lora_alpha, a.lora_dropout),
    )
    trainer.train()
    trainer.save_model(a.out)
    _dump_log(trainer, a.out)


def dpo(a):
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = load_rows(a.data)
    random.Random(0).shuffle(rows)
    ds = Dataset.from_list(
        [
            {
                "prompt": render_prompt(tok, [{"role": "user", "content": r["prompt"]}]),
                "chosen": render_completion(r["chosen"]),
                "rejected": render_completion(r["rejected"]),
            }
            for r in rows
        ]
    )
    cfg = DPOConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # transformers>=5: a float < 1 is a ratio
        beta=a.beta,
        # RPO term: a little NLL on the chosen response keeps DPO from pushing both
        # likelihoods down, which is a common cause of degenerate, rambling outputs.
        loss_type=["sigmoid", "sft"],
        loss_weights=[1.0, a.rpo_alpha],
        max_length=a.max_length,
        bf16=_has_cuda(),
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        max_steps=a.max_steps,
    )
    trainer = DPOTrainer(
        model=_load_model(a.model, a.grad_ckpt),
        ref_model=None,  # with LoRA, the reference is the same weights with the adapter disabled
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora(a.lora_r, a.lora_alpha, a.lora_dropout),
    )
    trainer.train()
    trainer.save_model(a.out)
    _dump_log(trainer, a.out)


def merge(a):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    base = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    model.save_pretrained(a.out, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
    try:
        AutoProcessor.from_pretrained(a.model).save_pretrained(a.out)
    except Exception as e:  # text-only test models have no processor
        print("[merge] no processor:", e)
    for name in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        src = Path(a.model) / name
        if src.exists():
            shutil.copy(src, Path(a.out) / name)


def _has_cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def _dump_log(trainer, out: str):
    Path(out, "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sft", "dpo", "merge"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data")
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rpo-alpha", type=float, default=0.2)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    a = ap.parse_args(argv)
    {"sft": sft, "dpo": dpo, "merge": merge}[a.stage](a)


if __name__ == "__main__":
    main()
