"""Mem-eff-only SDPA attention interface for Gemma 4 on Blackwell.

Gemma 4 has head_dim=320, which:
- FLASH rejects (head_dim>256 cap).
- CUDNN_ATTENTION silently rejects on sm_120.
- MATH accepts but materializes (h, s, s) fp32 softmax → 74 GB at seq=32k.
- EFFICIENT_ATTENTION (torch's mem-eff CUTLASS kernel) is the only backend
  that runs in O(seq) memory at long context. Measured peak at seq=32k,
  head=8, hd=320: 1.6 GB.

The default SDPA dispatcher can route head_dim=320 to MATH and OOM the card.
We register `attn_implementation="sdpa_memeff"` which forces the mem-eff
backend via `torch.nn.attention.sdpa_kernel([EFFICIENT_ATTENTION])`. Behavior
is otherwise identical to HF's stock `sdpa_attention_forward` — same causal
mask handling, same GQA key/value repeat, same dropout.

Importing this module registers the interface as a side effect.
"""
from __future__ import annotations

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# Pin to mem-eff only. Listed for clarity even though the list contains a
# single backend — keeps the intent explicit if we ever add fallbacks.
_SDPA_PREF = [SDPBackend.EFFICIENT_ATTENTION]


def sdpa_memeff_attention_forward(
    module, query, key, value, attention_mask=None,
    dropout: float = 0.0, scaling: float | None = None,
    is_causal: bool | None = None, **kwargs,
):
    """Drop-in replacement for HF's `sdpa_attention_forward`.

    Forces EFFICIENT_ATTENTION; necessary for Gemma 4's head_dim=320 (FLASH /
    CUDNN reject, MATH OOMs at long seq).
    """
    from transformers.integrations.sdpa_attention import repeat_kv

    if kwargs.get("output_attentions", False):
        import warnings
        warnings.warn("sdpa_memeff does not support output_attentions=True; ignored.")

    # EFFICIENT_ATTENTION rejects dense GQA with num_heads mismatch — repeat
    # K/V to match query heads. Cheap at our shapes (~33 MB per projection at
    # seq=32k, broadcasting 2→8 heads).
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if is_causal is None:
        is_causal = attention_mask is None and query.size(2) > 1

    sdpa_kwargs = {}
    if attention_mask is not None and not is_causal:
        sdpa_kwargs["attn_mask"] = attention_mask[:, :, :, : key.size(-2)]
    if scaling is not None:
        sdpa_kwargs["scale"] = scaling

    with sdpa_kernel(_SDPA_PREF):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            dropout_p=dropout,
            is_causal=is_causal and attention_mask is None,
            **sdpa_kwargs,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def register() -> None:
    """Register `sdpa_memeff` in ALL_ATTENTION_FUNCTIONS. Idempotent."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register("sdpa_memeff", sdpa_memeff_attention_forward)


# Register on import so `attn_implementation="sdpa_memeff"` is resolvable
# before from_pretrained.
register()
