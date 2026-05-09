"""Verify `PackedSFTCollator` produces per-sample loss equivalent to non-packed.

The test:
  1. Load 3 short samples from the SFT dataset.
  2. Run each INDIVIDUALLY through Gemma 4 → per-sample loss baseline.
  3. Feed the same 3 samples to `PackedSFTCollator` → packed batch.
  4. Run the packed batch through Gemma 4 (sdpa_memeff backend respects the
     4D block-diagonal mask) → compute per-sample loss on their positions.
  5. Compare: packed per-sample loss should match the non-packed baseline
     within bf16 tolerance.

If loss diverges, the mask plumbing is broken and samples are
cross-contaminating. If it matches, packing is training-equivalent to
non-packed and we can use it for the 2-3x throughput boost.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_samples(data_path: Path, n: int = 3, max_len: int = 1024):
    """Grab `n` short samples from the mixed SFT JSONL."""
    import sft.gemma4_liger_patch  # noqa: F401 — registers sdpa_memeff
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
    samples = []
    with open(data_path) as f:
        for line in f:
            r = json.loads(line.strip())
            tools = r.get("tools")
            if isinstance(tools, str):
                tools = json.loads(tools) if tools else None
            ck = r.get("chat_template_kwargs") or {}
            msgs = list(r["prompt"]) + list(r["completion"])
            rendered = tok.apply_chat_template(
                msgs, tools=tools or None, tokenize=False,
                add_generation_prompt=False, **ck,
            )
            ids = tok(rendered, add_special_tokens=False)["input_ids"]
            prompt_rendered = tok.apply_chat_template(
                list(r["prompt"]), tools=tools or None, tokenize=False,
                add_generation_prompt=False, **ck,
            )
            prompt_ids = tok(prompt_rendered, add_special_tokens=False)["input_ids"]
            if len(ids) > max_len or len(ids) <= len(prompt_ids):
                continue
            labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
            samples.append({"input_ids": ids, "labels": labels,
                            "total_len": len(ids)})
            if len(samples) >= n:
                break
    return tok, samples


def per_sample_loss_nonpacked(model, samples, device):
    losses = []
    for s in samples:
        ids = torch.tensor([s["input_ids"]], device=device)
        lbl = torch.tensor([s["labels"]], device=device)
        with torch.no_grad():
            out = model.forward(input_ids=ids, labels=lbl, skip_logits=False)
        losses.append(float(out.loss))
    return losses


def per_sample_loss_packed(model, samples, device, tokenizer, max_length=2048):
    """Use our PackedSFTCollator to build one packed batch, run through model,
    compute per-sample CE loss on each sample's token range.
    """
    from sft.packing import PackedSFTCollator
    tok_pad = tokenizer.pad_token_id
    if tok_pad is None:
        tok_pad = tokenizer.eos_token_id or 0
    collator = PackedSFTCollator(max_length=max_length, pad_token_id=tok_pad)
    batch = collator([{"input_ids": s["input_ids"], "labels": s["labels"]}
                      for s in samples])
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        out = model.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],          # 2D pad mask
            packed_attention_mask=batch["packed_attention_mask"],  # 4D block-diag
            position_ids=batch["position_ids"],
            skip_logits=False,
        )
    logits = out.logits  # (B, T, V)

    # The collator packs all samples into a single bin (B=1 if they fit).
    # Extract each sample's token range using position_ids — every time
    # pos resets to 0 a new sample starts.
    pos = batch["position_ids"][0].tolist()
    boundaries = [0]
    for i in range(1, len(pos)):
        if pos[i] == 0 and pos[i - 1] != 0:
            boundaries.append(i)
    boundaries.append(len(pos))

    assert len(boundaries) - 1 == len(samples), (
        f"Expected {len(samples)} samples in pack, got {len(boundaries) - 1}"
    )

    # The collator may reorder samples (first-fit-decreasing). Map each packed
    # segment back to the original sample by matching length.
    segment_lens = [boundaries[i + 1] - boundaries[i] for i in range(len(samples))]
    sample_to_segment = {}
    used = set()
    for si, s in enumerate(samples):
        L = len(s["input_ids"])
        for seg_i, sl in enumerate(segment_lens):
            if seg_i not in used and sl == L:
                sample_to_segment[si] = seg_i
                used.add(seg_i)
                break

    # Standard next-token shift: loss for predicting token t+1 from hidden t.
    shift_logits = logits[0, :-1, :]
    shift_labels = batch["labels"][0, 1:]

    losses = []
    for i in range(len(samples)):
        seg = sample_to_segment[i]
        start = boundaries[seg]
        end = boundaries[seg + 1]
        # shift space is offset by 1 — sample i spans [start, end) in original,
        # which is [start, end-1) after the shift-by-1 slice (last position
        # predicts nothing within-sample).
        s_start = start
        s_end = end - 1
        if s_end <= s_start:
            losses.append(float("nan"))
            continue
        sel_logits = shift_logits[s_start:s_end]
        sel_labels = shift_labels[s_start:s_end]
        valid = sel_labels != -100
        if valid.sum() == 0:
            losses.append(float("nan"))
            continue
        loss = F.cross_entropy(sel_logits[valid].float(), sel_labels[valid])
        losses.append(float(loss))
    return losses


def main():
    device = "cuda"
    data_path = Path("sft/data/agent_dd_mixed.norm.jsonl")
    print(f"Loading samples from {data_path}...")
    tokenizer, samples = load_samples(data_path, n=3, max_len=800)
    print(f"  got {len(samples)} samples, lengths = {[s['total_len'] for s in samples]}")

    print("Loading model (sdpa_memeff)...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E4B-it", dtype=torch.bfloat16,
        device_map={"": device}, attn_implementation="sdpa_memeff",
    ).eval()
    # Install our fused-CE + packed_attention_mask swap forward.
    from sft.gemma4_liger_patch import apply_chunked_ce_to_gemma4
    apply_chunked_ce_to_gemma4(model)

    print("\n--- Non-packed baseline ---")
    lnp = per_sample_loss_nonpacked(model, samples, device)
    for i, l in enumerate(lnp):
        print(f"  sample[{i}] loss = {l:.6f}")

    print("\n--- Packed (PackedSFTCollator + block-diag mask) ---")
    lp = per_sample_loss_packed(model, samples, device, tokenizer,
                                max_length=sum(s["total_len"] for s in samples) + 16)
    for i, l in enumerate(lp):
        print(f"  sample[{i}] loss = {l:.6f}")

    # Tolerance: 2e-2. bf16 dtype alone costs ~1.6e-3 on CE; deep-network
    # accumulation pushes that up to ~1e-2. Cross-contamination would produce
    # order-of-magnitude larger deltas (observed 0.5-0.7 when the mask was
    # dropped during debugging).
    print("\n--- Comparison ---")
    all_pass = True
    for i in range(len(samples)):
        diff = abs(lnp[i] - lp[i])
        ok = diff < 2e-2
        print(f"  sample[{i}] diff = {diff:.6f}  {'PASS' if ok else 'FAIL'}")
        all_pass = all_pass and ok
    print(f"\n  OVERALL: {'PASS — packing is semantically equivalent' if all_pass else 'FAIL — cross-contamination detected'}")


if __name__ == "__main__":
    main()
