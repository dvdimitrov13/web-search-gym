"""SFT for the agent_dd browse_page extractor using Unsloth on Qwen3.5-2B.

Single-task page→bullets extraction, isolated from the searcher stage.
Train on `sft/data/extractor_v2_train.jsonl`, eval loss on `extractor_v2_val.jsonl`.

Why 2B + packing: profile_train.py / probe_2b_32k.py showed 4B at seq=32k
fits only with Unsloth's CPU-offload grad checkpointing, leaving the GPU at
~50% util on PCIe stalls. 2B fits seq=32k natively (22 GB peak, 10 GB free)
with plain checkpointing — GPU runs at ~100%. Sample packing (`packing=True`)
amortizes the 5-12s fixed CPU launch overhead across ~20 short samples per
32k window (median sample = ~1.5k tokens), where short-sample training was
otherwise launch-bound on a 4B model.

Invocation:
    uv run python -m sft.train_unsloth_extractor --config sft/configs/extractor_v2_qwen35.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _wrap_text_blocks(messages: list[dict]) -> list[dict]:
    """Convert each message's string content into a `[{type:text, text:...}]` list.

    Qwen3.5 is a multimodal model — its chat template iterates `content` looking
    for image/video blocks. When `content` is a plain string the iteration yields
    individual characters and crashes with `TypeError: string indices must be
    integers`. Wrapping the text into the multimodal block format keeps the
    template happy without changing the rendered prompt.
    """
    out = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            m = {**m, "content": [{"type": "text", "text": c}]}
        out.append(m)
    return out


def _load_dataset(path: Path):
    """Load SFT JSONL (prompt + completion + optional tools per row).

    Keeps `tools` so multi-turn searcher data renders tool defs in the prompt.
    No content-wrap: we pass the unwrapped text tokenizer to TRL (not the
    multimodal processor), so plain string content tokenizes correctly without
    the [{type:text, text:...}] wrapping that the multimodal template needs.
    """
    # Use Dataset.from_list to bypass pyarrow schema inference. Searcher rows
    # have variable-length message arrays (turn_idx=0 has 2 messages, turn_idx=N
    # has 2+2N messages with tool_calls/tool roles), which load_dataset("json")
    # can't unify into a single arrow schema.
    from datasets import Dataset
    keep = ("prompt", "completion", "chat_template_kwargs", "tools")
    records = []
    with open(path) as src:
        for line in src:
            r = json.loads(line)
            rec = {k: v for k, v in r.items() if k in keep}
            rec.setdefault("chat_template_kwargs", {"enable_thinking": False})
            records.append(rec)
    return Dataset.from_list(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Cap optimizer steps for profiling runs; -1 = run full epochs.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    max_seq = int(train_cfg["max_seq_length"])

    # Unsloth must be imported BEFORE transformers — it monkey-patches model classes.
    from unsloth import FastLanguageModel

    print(f"Loading {model_cfg['name']} at seq={max_seq} via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=max_seq,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    # Qwen3.5's "tokenizer" returned by Unsloth is actually a Qwen3VLProcessor
    # (multimodal ProcessorMixin). Both TRL's _is_vlm check and Unsloth's
    # compiled-cache check trip on this and silently disable packing. Our data
    # is text-only, so swap in the underlying text tokenizer — it carries the
    # same chat template and tokenization behavior without the VLM flag.
    from transformers import ProcessorMixin
    if isinstance(tokenizer, ProcessorMixin):
        inner_tok = getattr(tokenizer, "tokenizer", None)
        if inner_tok is None:
            raise RuntimeError("processor has no .tokenizer attribute; cannot unwrap")
        print(f"Unwrapped processor → tokenizer ({type(inner_tok).__name__})")
        tokenizer = inner_tok
    print("Wrapping with LoRA via Unsloth...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        use_gradient_checkpointing=True,
        random_state=train_cfg.get("seed", 42),
        max_seq_length=max_seq,
    )

    train_ds = _load_dataset(Path(data_cfg["train"]))
    val_ds = _load_dataset(Path(data_cfg["val"])) if data_cfg.get("val") else None
    print(f"Train: {len(train_ds)} samples; Val: {len(val_ds) if val_ds else 0} samples")

    from trl import SFTConfig, SFTTrainer

    out_dir = train_cfg.get("output_dir", "sft/checkpoints/extractor_v2_qwen35")
    sft_args = SFTConfig(
        output_dir=out_dir,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=train_cfg.get("epochs", 2),
        per_device_train_batch_size=train_cfg.get("batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation", 8),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        max_length=max_seq,
        logging_steps=train_cfg.get("logging_steps", 5),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=train_cfg.get("eval_steps", 50),
        per_device_eval_batch_size=1,
        bf16=True,
        # For prompt+completion datasets, fire loss only on completion tokens.
        completion_only_loss=True,
        packing=bool(train_cfg.get("packing", False)),
        remove_unused_columns=False,
        report_to="wandb" if train_cfg.get("wandb_project") else "none",
        seed=train_cfg.get("seed", 42),
        eval_on_start=bool(train_cfg.get("eval_on_start", False)),
    )

    if train_cfg.get("wandb_project"):
        import os
        os.environ.setdefault("WANDB_PROJECT", train_cfg["wandb_project"])
        if train_cfg.get("wandb_run_name"):
            os.environ.setdefault("WANDB_NAME", train_cfg["wandb_run_name"])

    # Exempt specific milestone steps from save_total_limit rotation in place
    # (no rename, no copy) by filtering them out of the rotation candidate set.
    # End-of-epoch saves are appended at runtime by _EpochEndSavePin so we don't
    # need to know their step numbers in advance.
    keep_steps: set[int] = {int(s) for s in train_cfg.get("keep_checkpoint_steps", [])}

    class _SFTTrainerKeepMilestones(SFTTrainer):
        def _rotate_checkpoints(self, use_mtime=False, output_dir=None):
            if not self.args.save_total_limit or self.args.save_total_limit <= 0:
                return
            import re, shutil
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            re_ckpt = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"-(\d+)$")
            ckpts = []
            for p in Path(output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*"):
                m = re_ckpt.match(p.name)
                if not (m and p.is_dir()):
                    continue
                step = int(m.group(1))
                if step in keep_steps:
                    continue
                ckpts.append((step, str(p)))
            ckpts.sort()
            n_to_delete = max(0, len(ckpts) - self.args.save_total_limit)
            for _, path in ckpts[:n_to_delete]:
                shutil.rmtree(path, ignore_errors=True)
                print(f"[Rotate] Deleted {path}")

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            # Bypass the full forward + lm_head — that materializes a (B, T, V=248064)
            # bf16 logits tensor (16.25 GB at seq=32k) which OOMs the 24 GB card.
            # Unsloth gates its Liger fused-CE patch on self.training, leaving eval to
            # take the slow path. We replicate it here: backbone -> hidden states ->
            # LigerFusedLinearCrossEntropyLoss, which chunks across the vocab dim.
            import torch
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )
            causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
            with torch.no_grad():
                outputs = causal_lm.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    use_cache=False,
                )
                hidden = outputs.last_hidden_state  # (B, T, H)
                labels = inputs["labels"]
                shift_hidden = hidden[..., :-1, :].contiguous().view(-1, hidden.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                loss_fn = LigerFusedLinearCrossEntropyLoss()
                loss = loss_fn(causal_lm.lm_head.weight, shift_hidden, shift_labels)
            return (loss.detach(), None, None)

    from transformers import TrainerCallback

    class _EpochEndSavePin(TrainerCallback):
        # Force a checkpoint at end of every epoch AND pin its step against
        # save_total_limit rotation. Mutates keep_steps in the enclosing scope so
        # _SFTTrainerKeepMilestones._rotate_checkpoints sees the new pin.
        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = True
            keep_steps.add(int(state.global_step))
            print(f"[EpochEndSavePin] Pinned step {state.global_step} (end of epoch)")

    trainer = _SFTTrainerKeepMilestones(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[_EpochEndSavePin()],
    )

    print(f"\nStart training: epochs={sft_args.num_train_epochs}, "
          f"eff batch={sft_args.per_device_train_batch_size}*{sft_args.gradient_accumulation_steps}={sft_args.per_device_train_batch_size * sft_args.gradient_accumulation_steps}, "
          f"lr={sft_args.learning_rate}")
    trainer.train()

    save_dir = train_cfg.get("save_dir", "sft/output/extractor_v2_qwen35")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved → {save_dir}")


if __name__ == "__main__":
    main()
