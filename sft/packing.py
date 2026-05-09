"""Custom packed-sequence collator for Gemma 4 SFT.

TRL's native `packing=True, packing_strategy="bfd"` wires into flash-attn's
varlen API via `cu_seqlens`. We don't have flash-attn (build fails for
sm_120), so we can't use that path. Instead we produce a 4D block-diagonal
attention mask that our `sdpa_memeff` attention interface consumes directly
— no padding waste, samples stay fully independent in attention.

Usage:
    from sft.packing import PackedSFTCollator
    trainer = SFTTrainer(
        ...,
        data_collator=PackedSFTCollator(max_length=24576, pad_token_id=tok.pad_token_id),
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class PackedSFTCollator:
    """Greedy first-fit packer + block-diagonal attention mask builder.

    Each call consumes a list of tokenized samples (dicts with `input_ids`,
    `labels`) and emits one batch of packed sequences where:
      - `input_ids` has multiple samples concatenated, pad-filled at the tail.
      - `labels` matches `input_ids` but with -100 everywhere the sample's
        original label was -100 (prompt tokens) and on pad positions.
      - `attention_mask` is a 4D bool tensor (B, 1, T, T) where
        `attn[b, 0, i, j]` is True iff token j belongs to the same sample as
        token i AND j ≤ i (causal within each sample).
      - `position_ids` reset to 0 at each sample boundary so RoPE/absolute
        embeddings line up correctly.

    Note: this is a *drop-in* replacement for TRL's default collator — it
    works with the usual dataset shape (`input_ids` + `labels` keys, produced
    by TRL's tokenize_fn).
    """

    max_length: int
    pad_token_id: int
    # Ignore index for label padding (standard HF CE default).
    label_pad_token_id: int = -100
    # If a single sample exceeds max_length, truncate its tail (the prompt
    # tail). Completion is preserved because it comes at the end.
    truncate_overflowing: bool = True

    def __call__(self, features: Iterable[dict]) -> dict:
        samples = []
        for f in features:
            ids = f["input_ids"]
            lbl = f.get("labels", ids)
            # Accept python lists or tensors — normalize to python lists.
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if hasattr(lbl, "tolist"):
                lbl = lbl.tolist()
            # Safety: truncate single samples that exceed the packing budget.
            if len(ids) > self.max_length and self.truncate_overflowing:
                ids = ids[-self.max_length:]
                lbl = lbl[-self.max_length:]
            samples.append((ids, lbl))

        # Greedy first-fit packing into bins of size max_length.
        bins: list[list[tuple[list[int], list[int]]]] = []
        bin_lens: list[int] = []
        # Sort by length descending for tighter packing (first-fit decreasing).
        order = sorted(range(len(samples)), key=lambda i: -len(samples[i][0]))
        for i in order:
            ids, lbl = samples[i]
            placed = False
            for b, blen in enumerate(bin_lens):
                if blen + len(ids) <= self.max_length:
                    bins[b].append((ids, lbl))
                    bin_lens[b] = blen + len(ids)
                    placed = True
                    break
            if not placed:
                bins.append([(ids, lbl)])
                bin_lens.append(len(ids))

        B = len(bins)
        T = max(bin_lens) if bin_lens else 1

        input_ids = torch.full((B, T), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, T), self.label_pad_token_id, dtype=torch.long)
        position_ids = torch.zeros((B, T), dtype=torch.long)
        # 2D padding mask: 1 where a real token sits, 0 where pad. TRL's
        # entropy metric multiplies per_token_entropy (B,T) by this; must be 2D.
        attn_mask_2d = torch.zeros((B, T), dtype=torch.long)
        # 4D block-diagonal attention mask. Carried under a SEPARATE key so
        # TRL-side metric code sees only the 2D pad mask. Our
        # `patched_forward` pops this and threads it into the inner model.
        # bf16 to match SDPA kernel dtype downstream.
        packed_attn_mask = torch.full(
            (B, 1, T, T), float("-inf"), dtype=torch.bfloat16,
        )

        for b, pack in enumerate(bins):
            offset = 0
            # Build sample_id vector for this pack's occupied positions.
            sids = []
            for sid, (ids, lbl) in enumerate(pack):
                L = len(ids)
                input_ids[b, offset:offset + L] = torch.tensor(ids, dtype=torch.long)
                labels[b, offset:offset + L] = torch.tensor(lbl, dtype=torch.long)
                position_ids[b, offset:offset + L] = torch.arange(L, dtype=torch.long)
                attn_mask_2d[b, offset:offset + L] = 1
                sids += [sid] * L
                offset += L
            occ = offset  # number of occupied positions in this bin
            # Pad positions get sid = -1 so they never match anyone — mask them out.
            sids += [-1] * (T - occ)
            sid_vec = torch.tensor(sids, dtype=torch.long)
            same = sid_vec.unsqueeze(0) == sid_vec.unsqueeze(1)  # (T, T) bool
            same = same & (sid_vec.unsqueeze(0) >= 0) & (sid_vec.unsqueeze(1) >= 0)
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
            allow = same & causal  # (T, T) bool
            # Convert to float: 0.0 where allowed, -inf elsewhere.
            packed_attn_mask[b, 0] = torch.where(
                allow,
                torch.tensor(0.0, dtype=torch.bfloat16),
                torch.tensor(float("-inf"), dtype=torch.bfloat16),
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn_mask_2d,
            "packed_attention_mask": packed_attn_mask,
            "position_ids": position_ids,
        }
