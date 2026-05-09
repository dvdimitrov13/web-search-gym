"""Monkey-patch Gemma 4 with Liger fused linear-CE.

Liger-Kernel 0.7.0 supports gemma/gemma2/gemma3 but not gemma4. Gemma 4's
vocab=262,144 makes the (B,T,V) logits tensor ~17 GB at T=32k in bf16 and
another 34 GB once the loss upcasts it to fp32, OOMing a 96 GB card.

The fused kernel computes loss in fp32 chunks without materializing (B,T,V).
Pattern mirrors `liger_kernel.transformers.model.gemma3.causal_forward`,
adapted for Gemma 4's two model classes (text-only `Gemma4ForCausalLM` and
multimodal `Gemma4ForConditionalGeneration`) with one shared loss/logits
body.

Importing this module also registers the `sdpa_memeff` attention interface
(see `sft.sdpa_memeff`).
"""
from __future__ import annotations

from typing import Optional, Union

import torch
from liger_kernel.transformers.model.loss_utils import (
    LigerForCausalLMLoss,
    unpack_cross_entropy_result,
)
from torch.nn.attention import sdpa_kernel
from transformers.cache_utils import Cache
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4CausalLMOutputWithPast,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
)

# Importing for the registration side-effect AND to share the SDPA backend
# preference with our patched forwards (so the inner `self.model(...)` runs
# under the same mem-eff pin we install at the attention-interface level).
from sft.sdpa_memeff import _SDPA_PREF  # noqa: F401 — re-exported as a constant


def _compute_loss_and_logits(
    self,
    inner_outputs,
    *,
    labels: Optional[torch.LongTensor],
    logits_to_keep: Union[int, torch.Tensor],
    skip_logits: Optional[bool],
    shift_labels: Optional[torch.Tensor],
    extra_loss_kwargs: dict,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Shared body of both patched forwards.

    Returns (loss, logits). `logits` is the full lm_head output in eval, or a
    `(B, T, 1)` zeros tensor in the skip_logits training path — the latter
    keeps TRL's `entropy_from_logits(outputs.logits)` call happy without
    materializing the full vocab projection (which is the OOM we're fixing).
    """
    hidden_states = inner_outputs.last_hidden_state
    slice_indices = (
        slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    )
    kept_hidden_states = hidden_states[:, slice_indices, :]

    text_cfg = self.config.get_text_config()
    softcap = getattr(text_cfg, "final_logit_softcapping", None)

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    if skip_logits:
        result = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            hidden_size=text_cfg.hidden_size,
            shift_labels=shift_labels,
            final_logit_softcapping=softcap,
            **extra_loss_kwargs,
        )
        loss, _, _ = unpack_cross_entropy_result(result)
        # TRL's SFTTrainer.compute_loss reads outputs.logits unconditionally
        # for the entropy metric. Dummy (B, T, 1) zeros: shape/attribute
        # access works, gradient comes from `loss` above.
        logits = kept_hidden_states.new_zeros(
            kept_hidden_states.size(0), kept_hidden_states.size(1), 1
        )
        return loss, logits

    # Eval / generation: preserve upstream behavior.
    logits = self.lm_head(kept_hidden_states)
    if softcap is not None:
        logits = logits / softcap
        logits = torch.tanh(logits)
        logits = logits * softcap
    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, text_cfg.vocab_size)
    return loss, logits


def patched_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    input_features_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    image_position_ids: Optional[torch.LongTensor] = None,
    video_position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    mm_token_type_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **kwargs,
) -> Gemma4CausalLMOutputWithPast:
    """Patched forward for the multimodal `Gemma4ForConditionalGeneration`."""
    # Trainer may inject return_dict; force the dict shape.
    kwargs.pop("return_dict", None)
    # PackedSFTCollator emits a 4D block-diagonal mask under
    # `packed_attention_mask`; swap it into attention_mask for the inner
    # model call so attention sees real per-sample boundaries.
    packed_mask = kwargs.pop("packed_attention_mask", None)
    inner_attention_mask = packed_mask if packed_mask is not None else attention_mask
    shift_labels = kwargs.pop("shift_labels", None)

    # Pin SDPA to mem-eff for the entire forward; `sdpa_memeff` already does
    # this at the attention-interface level, but the model's own pre/post
    # attention work can also dispatch SDPA, and head_dim=320 to MATH OOMs.
    with sdpa_kernel(_SDPA_PREF):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=inner_attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            return_dict=True,
            **kwargs,
        )

    extra = {k: v for k, v in kwargs.items() if k in ("num_items_in_batch",)}
    loss, logits = _compute_loss_and_logits(
        self,
        outputs,
        labels=labels,
        logits_to_keep=logits_to_keep,
        skip_logits=skip_logits,
        shift_labels=shift_labels,
        extra_loss_kwargs=extra,
    )

    return Gemma4CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        audio_hidden_states=outputs.audio_hidden_states,
    )


def patched_forward_causal_lm(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **kwargs,
) -> Gemma4CausalLMOutputWithPast:
    """Patched forward for the text-only `Gemma4ForCausalLM`.

    Same loss/logits body as `patched_forward`; differs only in the inner
    model signature (no multimodal kwargs) and that the output's
    image/audio fields are absent (default to None on the dataclass).
    """
    kwargs.pop("return_dict", None)
    packed_mask = kwargs.pop("packed_attention_mask", None)
    inner_attention_mask = packed_mask if packed_mask is not None else attention_mask
    shift_labels = kwargs.pop("shift_labels", None)

    with sdpa_kernel(_SDPA_PREF):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=inner_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

    extra = {k: v for k, v in kwargs.items() if k in ("num_items_in_batch",)}
    loss, logits = _compute_loss_and_logits(
        self,
        outputs,
        labels=labels,
        logits_to_keep=logits_to_keep,
        skip_logits=skip_logits,
        shift_labels=shift_labels,
        extra_loss_kwargs=extra,
    )

    return Gemma4CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=getattr(outputs, "past_key_values", None),
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
    )


def _unwrap_to_gemma4(obj):
    """Return the underlying Gemma 4 model (either variant) from a PEFT wrapper."""
    seen = set()
    cur = obj
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (Gemma4ForConditionalGeneration, Gemma4ForCausalLM)):
            return cur
        nxt = getattr(cur, "base_model", None)
        if nxt is not None and nxt is not cur:
            inner = getattr(nxt, "model", nxt)
            cur = inner
            continue
        cur = getattr(cur, "module", None)
    return None


def apply_chunked_ce_to_gemma4(model) -> None:
    """Install the Liger fused-CE forward on the underlying Gemma 4 model.

    Dispatches on class so the multimodal vs text-only signature differences
    are preserved (HF Trainer / TRL reflect on `forward` to decide what to
    pass; using a `**kwargs`-only signature breaks that).

    The `_resolve_lm_head_weight` PEFT-aware branch we used to carry was
    dropped — `lm_head` is intentionally kept out of LoRA targets in our
    training configs, so `lm_head.weight` is always the right reference.
    """
    target = _unwrap_to_gemma4(model)
    if target is None:
        raise TypeError(
            f"apply_chunked_ce_to_gemma4: no Gemma4ForCausalLM / "
            f"Gemma4ForConditionalGeneration under {type(model).__name__}"
        )
    fn = (
        patched_forward_causal_lm
        if isinstance(target, Gemma4ForCausalLM)
        else patched_forward
    )
    target.forward = fn.__get__(target, target.__class__)
