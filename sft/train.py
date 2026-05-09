"""SFT training for Qwen3 / Gemma 4 on converted traces.

Straight TRL + LoRA. `assistant_only_loss=True` means only assistant tokens
contribute to the loss; tool results and user/system prompts are masked out.

Invocation:
    uv run python -m sft.train --data sft/data/<converted>.jsonl --config sft/configs/smoke.yaml
    uv run python -m sft.train --data sft/data/<converted>.jsonl --config sft/configs/cloud.yaml

Two configs ship: `smoke.yaml` (Qwen3-0.6B local on M-series MPS, ~3 steps,
verifies the pipeline runs) and `cloud.yaml` (Gemma 4 E4B-it LoRA on a
vast.ai 5090 / RTX PRO 6000, real training).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "smoke.yaml"


def _load_dataset(path: Path):
    """Load a converted SFT JSONL with heterogeneous tool_call schemas.

    Each assistant turn may carry `tool_calls[i].function.arguments` with a
    tool-specific dict shape (e.g. commit_memory.new_text vs answer.citations).
    `Dataset.from_list` routes through pyarrow's strict struct-merge, which
    fails on cross-row struct heterogeneity. Writing a normalized JSONL to
    disk and loading via `load_dataset('json', ...)` uses the JSON reader
    path, which tolerates this by widening each call site's arguments to a
    struct with the union of keys (missing fields become nulls).

    We serialize `tools` to a JSON string (TRL's tokenize_fn json.loads it)
    so searcher (has tools) and extractor (no tools) samples stay uniform.
    """
    from datasets import load_dataset

    allowed = ("messages", "prompt", "completion", "tools", "chat_template_kwargs")
    norm_path = path.with_suffix(".norm.jsonl")
    with open(path) as src, open(norm_path, "w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rec = {k: v for k, v in r.items() if k in allowed}
            # Keep `tools` as a native list (transformers 5.x rejects JSON strings).
            # Also rehydrate if an upstream stage pre-serialized it to a string.
            t = rec.get("tools")
            if isinstance(t, str):
                t = json.loads(t)
            if not t:
                rec.pop("tools", None)
            else:
                rec["tools"] = t
            rec.setdefault("chat_template_kwargs", {})
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return load_dataset("json", data_files=str(norm_path), split="train")


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

    # Confirmed via runtime test: on torch 2.11/Blackwell, both FLASH and
    # MEM_EFFICIENT SDPA backends reject Gemma 4's head_dim=320 — disabling
    # math forces "Invalid backend" at runtime. We're stuck with math (fp32
    # softmax, O(seq^2)), which caps trainable seq on a 96 GB card to ~22k.
    # Left this block in place so leaving everything enabled is explicit.
    if device == "cuda":
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(True)
        except Exception as e:
            print(f"[sdpa] backend hint failed: {e}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if train_cfg.get("bf16", False) else torch.float32
    attn_impl = train_cfg.get("attn_implementation")
    model_kwargs = {"dtype": dtype}
    if device != "cpu":
        model_kwargs["device_map"] = {"": device}
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    # For Gemma 4 we must load via AutoModelForCausalLM (picks the multimodal
    # Gemma4ForConditionalGeneration class for the E4B-it checkpoint). The
    # text-only Gemma4ForCausalLM class expects `model.*` key prefixes but the
    # checkpoint stores them as `model.language_model.*`, so using that class
    # leaves all weights MISSING and silently trains from random init.
    # Pre-register our `sdpa_memeff` attention interface BEFORE from_pretrained
    # so the attn_implementation string resolves at load time.
    if model_name.lower().startswith("google/gemma-4"):
        import sft.gemma4_liger_patch  # noqa: F401  — registers sdpa_memeff
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    peft_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        # Keep lm_head out of LoRA: our fused-CE patch feeds lm_head.weight
        # directly into Liger's custom autograd; a LoRA-wrapped lm_head would
        # orphan adapter grads. Also: lm_head is tied to embed_tokens, so
        # training it via a LoRA delta wouldn't do what you'd expect anyway.
        exclude_modules=lora_cfg.get("exclude_modules", ["lm_head"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Patch Gemma 4's forward to use Liger fused linear CE — avoids materializing
    # the (B, T, 262k) logits tensor. Required to train at seq≥16k on one 96 GB
    # card; Liger 0.7.0 doesn't register gemma4 natively.
    if cfg["model"]["name"].lower().startswith("google/gemma-4"):
        from sft.gemma4_liger_patch import apply_chunked_ce_to_gemma4
        apply_chunked_ce_to_gemma4(model)
        print("[patch] Applied Liger fused-CE forward to Gemma 4 model.")
    # LoRA + gradient checkpointing gotcha: without input_require_grads, the
    # frozen base weights produce no grad and GC stores full activations,
    # defeating the memory savings. Also force non-reentrant GC — the
    # reentrant path has a known interaction with LoRA that can duplicate
    # activation storage.
    if train_cfg.get("gradient_checkpointing"):
        model.enable_input_require_grads()

    dataset = _load_dataset(args.data).shuffle(seed=42)
    is_prompt_completion = "prompt" in dataset.column_names
    loss_mode = "completion_only" if is_prompt_completion else "assistant_only"
    print(f"Dataset: {len(dataset)} examples ({loss_mode} loss)")

    if train_cfg.get("drop_too_long"):
        max_len = int(train_cfg.get("max_seq_length", 8192))
        before = len(dataset)

        def _fits(ex):
            tools = ex.get("tools") or None
            ck = ex.get("chat_template_kwargs") or {}
            if is_prompt_completion:
                msgs = list(ex["prompt"]) + list(ex["completion"])
            else:
                msgs = ex["messages"]
            # apply_chat_template(tokenize=True) returns a dict on recent
            # transformers (input_ids + attention_mask). Render to a string and
            # tokenize manually for a reliable length count.
            rendered = tokenizer.apply_chat_template(
                msgs, tools=tools, tokenize=False,
                add_generation_prompt=False, **ck,
            )
            ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            return len(ids) <= max_len

        dataset = dataset.filter(_fits, num_proc=4)
        print(f"Pre-filter (max_seq={max_len}): {before} → {len(dataset)} "
              f"(dropped {before - len(dataset)})")

    sft_kwargs = {}
    if train_cfg.get("max_steps"):
        sft_kwargs["max_steps"] = int(train_cfg["max_steps"])
    if train_cfg.get("use_liger_kernel"):
        sft_kwargs["use_liger_kernel"] = True

    wandb_project = train_cfg.get("wandb_project") or None
    report_to = "wandb" if wandb_project else "none"
    if wandb_project:
        import os as _os
        _os.environ.setdefault("WANDB_PROJECT", wandb_project)
        if train_cfg.get("wandb_run_name"):
            _os.environ.setdefault("WANDB_NAME", train_cfg["wandb_run_name"])

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
        save_total_limit=train_cfg.get("save_total_limit"),
        bf16=train_cfg.get("bf16", False),
        fp16=False,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False} if train_cfg.get("gradient_checkpointing") else None,
        # `messages` datasets (whole mode): loss on every assistant token.
        # `prompt`+`completion` datasets (per-turn mode): TRL v1.2 default for
        # completion_only_loss is None — set explicitly so only the completion
        # tokens contribute loss. Without this, ALL labels end up as -100 and
        # loss becomes NaN (observed on TRL 1.2.0).
        assistant_only_loss=not is_prompt_completion,
        completion_only_loss=is_prompt_completion,
        packing=False,
        remove_unused_columns=False,
        report_to=report_to,
        seed=42,
        **sft_kwargs,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }
    if train_cfg.get("custom_packing"):
        # Replace TRL's default collator with our packed-sequence collator.
        # Requires an attn_implementation that respects 4D attention masks
        # (sdpa_memeff does).
        from sft.packing import PackedSFTCollator
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        trainer_kwargs["data_collator"] = PackedSFTCollator(
            max_length=int(train_cfg.get("max_seq_length", 8192)),
            pad_token_id=pad_id,
        )
        print("[packing] enabled custom PackedSFTCollator (block-diag mask)")
    trainer = SFTTrainer(**trainer_kwargs)

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
