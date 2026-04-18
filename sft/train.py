"""SFT training for Qwen3 on converted traces.

Straight TRL + LoRA. `assistant_only_loss=True` means only assistant tokens
contribute to the loss; tool results and user/system prompts are masked out.

Invocation:
    uv run python -m sft.train --data sft/data/<converted>.jsonl --config sft/configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "config.yaml"


def _load_dataset(path: Path):
    from datasets import Dataset

    allowed = ("messages", "prompt", "completion", "tools")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append({k: v for k, v in r.items() if k in allowed})
    # TRL v1.0 handles list-of-dicts fields natively. It auto-detects the
    # dataset format from columns: `messages` → conversational language-modeling
    # (assistant_only_loss), `prompt`+`completion` → prompt-completion
    # (completion_only_loss).
    return Dataset.from_list(records)


def _detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    import torch
    import yaml
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    device = _detect_device()
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    print(f"Device: {device}\nModel: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if train_cfg.get("bf16", False) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map={"": device} if device != "cpu" else None,
    )

    peft_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = _load_dataset(args.data).shuffle(seed=42)
    is_prompt_completion = "prompt" in dataset.column_names
    loss_mode = "completion_only" if is_prompt_completion else "assistant_only"
    print(f"Dataset: {len(dataset)} examples ({loss_mode} loss)")

    training_args = SFTConfig(
        output_dir=train_cfg.get("output_dir", "sft/checkpoints"),
        num_train_epochs=train_cfg.get("epochs", 1),
        per_device_train_batch_size=train_cfg.get("batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_length=train_cfg.get("max_seq_length", 8192),
        logging_steps=train_cfg.get("logging_steps", 1),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 30),
        bf16=train_cfg.get("bf16", False),
        fp16=False,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        # `messages` datasets (whole mode): loss on every assistant token.
        # `prompt`+`completion` datasets (per-turn mode): TRL defaults to
        # completion_only_loss, so only the final turn contributes loss.
        assistant_only_loss=not is_prompt_completion,
        packing=False,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(
        f"\nTraining: epochs={training_args.num_train_epochs}, "
        f"batch={training_args.per_device_train_batch_size}, "
        f"grad_accum={training_args.gradient_accumulation_steps}, "
        f"lr={training_args.learning_rate}"
    )
    trainer.train()

    save_dir = train_cfg.get("save_dir", "sft/output")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved → {save_dir}")


if __name__ == "__main__":
    main()
