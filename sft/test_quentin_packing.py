"""Reproduce @qgallouedec's TRL #5316 packing test against our PackedSFTCollator.

Quentin's test (https://github.com/huggingface/trl/pull/5316) showed that with
SmolLM2-135M and 3 packed sequences, only flash-attn2 produces per-sample
logits matching the non-packed reference. `sdpa`/`flex_attention`/`eager` fail
with max_diff 12-14 on all but the first sample — clear cross-sample
attention bleed.

His method: flatten sequences into one tensor, reset `position_ids` at each
boundary, pass NO attention mask. That's insufficient — models need an
explicit 4D block-diagonal mask to block cross-sample attention.

This script runs THREE variants:

1. **No mask** (Quentin's test, negative control) — expected to fail on seq[1+].
2. **Our PackedSFTCollator 4D mask** — expected to pass.
3. **Stock 2D pad mask** (what TRL emits today without our collator) — also expected to fail.

If variant 2 passes with `attn_implementation="sdpa"`, that's concrete evidence
that TRL #5316 needs a mask-emitting collator — not just a whitelist change.
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM

from sft.packing import PackedSFTCollator


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def run_variant(name: str, model, seqs: list[torch.Tensor],
                attention_mask: torch.Tensor | None,
                position_ids: torch.Tensor | None,
                packed_ordering: list[int] | None = None) -> bool:
    """Run one packing variant, compare to per-sequence reference.

    packed_ordering: for each packed segment index, which original seq is it?
                     (FFD reorders; for plain no-mask tests it's [0, 1, 2, ...])
    """
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    seq_lengths = [s.shape[1] for s in seqs]

    with torch.no_grad():
        refs = [model(input_ids=s).logits for s in seqs]

        flat_ids = torch.cat([seqs[i] for i in packed_ordering], dim=1)
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        flat_out = model(input_ids=flat_ids, **kwargs).logits

        packed_lens = [seq_lengths[i] for i in packed_ordering]
        splits = flat_out.split(packed_lens, dim=1)

        all_pass = True
        for packed_i, seg in enumerate(splits):
            orig_i = packed_ordering[packed_i]
            ref = refs[orig_i]
            diff = (ref - seg).abs().max().item()
            ok = diff < 1e-3
            marker = "PASS" if ok else "FAIL"
            print(f"  seq[{orig_i}] (len={seq_lengths[orig_i]}): {marker}  max_diff={diff:.2e}")
            all_pass = all_pass and ok
    return all_pass


def main():
    print(f"Loading {MODEL_NAME} (fp32 CPU — correctness only)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, attn_implementation="sdpa",
    ).eval()

    torch.manual_seed(42)
    seqs = [
        torch.randint(100, 1000, (1, 6)),
        torch.randint(100, 1000, (1, 4)),
        torch.randint(100, 1000, (1, 7)),
    ]
    seq_lengths = [s.shape[1] for s in seqs]
    total = sum(seq_lengths)

    results = {}

    # --- Variant 1: Quentin's original test — no mask, position_ids only ---
    pos_ids_naive = torch.cat(
        [torch.arange(L) for L in seq_lengths]
    ).unsqueeze(0)
    results["V1 (no mask, pos_ids only)"] = run_variant(
        "V1: Quentin's test — no mask, position_ids only",
        model, seqs,
        attention_mask=None,
        position_ids=pos_ids_naive,
        packed_ordering=[0, 1, 2],
    )

    # --- Variant 2: our PackedSFTCollator 4D block-diag mask ---
    collator = PackedSFTCollator(max_length=total, pad_token_id=0)
    features = [{"input_ids": s[0].tolist()} for s in seqs]
    batch = collator(features)
    # Collator reorders via FFD (longest-first). Recover ordering by length.
    pos = batch["position_ids"][0].tolist()
    boundaries = [0] + [i for i in range(1, len(pos))
                        if pos[i] == 0 and pos[i - 1] != 0] + [len(pos)]
    seg_lens = [boundaries[i + 1] - boundaries[i] for i in range(len(seqs))]
    # Map packed segment index -> original seq index (match by length).
    used = set()
    packed_to_orig = []
    for sl in seg_lens:
        for orig_i, L in enumerate(seq_lengths):
            if orig_i not in used and L == sl:
                packed_to_orig.append(orig_i)
                used.add(orig_i)
                break
    assert len(packed_to_orig) == len(seqs)
    print(f"\n[info] FFD reordering: packed segments {seg_lens} -> original seqs {packed_to_orig}")

    # Pass the 4D mask DIRECTLY as attention_mask. Stock transformers models
    # that respect 4D masks (since #39194) consume it correctly.
    mask_4d = batch["packed_attention_mask"].to(torch.float32)  # cast for fp32 model

    results["V2 (our 4D block-diag mask)"] = run_variant(
        "V2: our PackedSFTCollator — 4D block-diag mask + pos_ids",
        model, seqs,
        attention_mask=mask_4d,
        position_ids=batch["position_ids"],
        packed_ordering=packed_to_orig,
    )

    # --- Variant 3: 2D pad mask only (what TRL today would emit for a packed batch) ---
    pad_mask_2d = batch["attention_mask"]
    results["V3 (2D pad mask only)"] = run_variant(
        "V3: 2D pad mask only (TRL default for packed batch w/o flash-attn)",
        model, seqs,
        attention_mask=pad_mask_2d,
        position_ids=batch["position_ids"],
        packed_ordering=packed_to_orig,
    )

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    # Interpretation: V1 should fail (Quentin showed this); V2 should pass
    # (our 4D mask); V3 should fail (2D pad mask lets all positions attend).
    if results["V2 (our 4D block-diag mask)"] and not results["V1 (no mask, pos_ids only)"]:
        print("\n✓ Our PackedSFTCollator is the missing piece for TRL #5316.")
        sys.exit(0)
    else:
        print("\n✗ Unexpected result — inspect per-sequence diffs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
