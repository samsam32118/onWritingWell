"""LoRA SFT and DPO for Qwen3.8 (any Qwen3.5-architecture model), and merging.

Both stages render the chat template themselves with thinking disabled, so training sees
exactly the tokens vLLM sees at inference:
    prompt     = <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
    completion = <response text><|im_end|>
LoRA touches only the language model's linear layers. `merge` writes a full checkpoint in
the original vision-language format so vLLM serves it unchanged.

    python -m owl.train sft   --model M --data sft.jsonl   --out ADAPTER [...]
    python -m owl.train dpo   --model M --data pairs.jsonl --out ADAPTER [...]
    python -m owl.train merge --model M --adapter ADAPTER  --out MODEL
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

LORA_TARGETS = (r".*language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))")


def train(a):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    random.Random(0).shuffle(rows)
    prompt = lambda msgs: tok.apply_chat_template(msgs, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    # Ends exactly at the EOS token so TRL does not append a second one.
    completion = lambda text: text.strip() + "<|im_end|>"
    common = dict(output_dir=a.out, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.05,
        max_length=a.max_length, bf16=torch.cuda.is_available(), logging_steps=a.logging_steps,
        save_strategy="no", report_to=[])
    if a.stage == "sft":
        ds = Dataset.from_list([{"prompt": prompt(r["messages"][:-1]),
            "completion": completion(r["messages"][-1]["content"])} for r in rows])
        args = SFTConfig(completion_only_loss=True, **common)
        Trainer = SFTTrainer
    else:
        ds = Dataset.from_list([{"prompt": prompt([{"role": "user", "content": r["prompt"]}]),
            "chosen": completion(r["chosen"]), "rejected": completion(r["rejected"])}
            for r in rows])
        # The SFT term on chosen (RPO) keeps DPO from pushing both likelihoods down,
        # a common cause of degenerate, rambling outputs. With LoRA the reference model
        # is the same weights with the adapter disabled.
        args = DPOConfig(beta=a.beta, loss_type=["sigmoid", "sft"],
            loss_weights=[1.0, a.rpo_alpha], **common)
        Trainer = DPOTrainer
    model = AutoModelForImageTextToText.from_pretrained(a.model, attn_implementation="sdpa",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    trainer = Trainer(model=model, args=args, train_dataset=ds, processing_class=tok,
        peft_config=lora)
    trainer.train()
    trainer.save_model(a.out)
    Path(a.out, "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))


def merge(a):
    from peft import PeftModel

    # On the GPU the merge takes seconds; on CPU it idles a billed GPU for many minutes.
    base = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else None)
    PeftModel.from_pretrained(base, a.adapter).merge_and_unload().save_pretrained(a.out,
        max_shard_size="5GB")
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
    for name in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json",
            "video_preprocessor_config.json"):
        if Path(a.model, name).exists():
            shutil.copy(Path(a.model, name), Path(a.out, name))


if __name__ == "__main__":
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
    ap.add_argument("--logging-steps", type=int, default=5)
    a = ap.parse_args()
    merge(a) if a.stage == "merge" else train(a)
