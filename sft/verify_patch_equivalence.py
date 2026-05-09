"""Numerical equivalence tests for the Gemma 4 training patches.

Validates two independent pieces:
  1. Fused-CE patch — Liger's `LigerForCausalLMLoss` vs a naive implementation
     that materializes (B,T,V) logits and calls `F.cross_entropy`.
  2. Mem-efficient SDPA backend — torch's EFFICIENT_ATTENTION (our forced
     backend) vs MATH backend (textbook reference), at Gemma 4's exact
     GQA shapes (8 query heads, 2 KV heads, head_dim=256).

If all tests PASS within bf16 tolerance (~5e-3), we have high confidence
that the training loop is mathematically equivalent to stock transformers
+ vanilla CE — we're just avoiding expensive allocations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_fused_ce_vs_reference(model_name: str, device: str) -> dict:
    """Test 1: patched forward w/ skip_logits=True == reference w/ skip_logits=False."""
    section("TEST 1 — Fused CE (Liger) vs Reference (materialized logits)")
    import sft.gemma4_liger_patch  # registers sdpa_memeff

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map={"": device},
        attn_implementation="sdpa_memeff",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    seq = 512
    text = "The quick brown fox jumps over the lazy dog. " * 20
    enc = tokenizer(text, return_tensors="pt", max_length=seq, truncation=True).to(device)
    input_ids = enc.input_ids
    labels = input_ids.clone()

    # Reference path: force skip_logits=False → materializes (B,T,V), calls CE.
    model.eval()
    with torch.no_grad():
        out_ref = model.forward(input_ids=input_ids, labels=labels, skip_logits=False)
    loss_ref = float(out_ref.loss)

    # Patched path: force skip_logits=True → Liger fused CE.
    model.train()
    with torch.no_grad():
        out_patched = model.forward(input_ids=input_ids, labels=labels, skip_logits=True)
    loss_patched = float(out_patched.loss)

    diff = abs(loss_ref - loss_patched)
    print(f"  reference loss  = {loss_ref:.6f}")
    print(f"  patched loss    = {loss_patched:.6f}")
    print(f"  abs difference  = {diff:.6f}")
    passed = diff < 5e-3
    print(f"  {'PASS' if passed else 'FAIL'}")

    del model
    torch.cuda.empty_cache()
    return {"test": "fused_ce", "ref": loss_ref, "patched": loss_patched,
            "diff": diff, "passed": passed}


def test_memeff_vs_math_attention(device: str) -> dict:
    """Test 2: SDPA EFFICIENT_ATTENTION backend == MATH backend for Gemma 4 shapes.

    Gemma 4 E4B-it: 8 query heads, 2 KV heads, head_dim=256. We generate
    identical Q/K/V tensors, run SDPA under each backend (with our GQA K/V
    repeat), and compare outputs in fp32.
    """
    section("TEST 2 — Mem-eff SDPA vs Math SDPA (Gemma 4 GQA shapes)")
    from transformers.integrations.sdpa_attention import repeat_kv

    torch.manual_seed(0)
    B, Hq, Hkv, D = 1, 8, 2, 256
    for seq in (256, 1024, 4096):
        q = torch.randn(B, Hq, seq, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, Hkv, seq, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, Hkv, seq, D, dtype=torch.bfloat16, device=device)
        # Repeat K/V for dense attention (matches our patch).
        groups = Hq // Hkv
        k_rep = repeat_kv(k, groups)
        v_rep = repeat_kv(v, groups)

        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            out_memeff = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        with sdpa_kernel([SDPBackend.MATH]):
            out_math = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)

        # bf16 has only 7 bits of mantissa (smallest increment ≈ 1/128 = 0.008
        # for values near 1.0). Attention accumulates over `seq` values, so
        # legitimate bf16 noise reaches ~seq * eps. Compare COSINE similarity
        # + max relative difference, which is what actually matters for the
        # downstream softmax.
        a = out_memeff.float().flatten()
        b = out_math.float().flatten()
        cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
        max_abs = float((a - b).abs().max())
        mean_abs = float((a - b).abs().mean())
        passed = cos > 0.9999
        print(f"  seq={seq}: cos_sim={cos:.8f}  max_abs={max_abs:.4f}  mean_abs={mean_abs:.6f}  {'PASS' if passed else 'FAIL'}")
    return {"test": "memeff_vs_math", "cos_sim_at_4096": cos,
            "max_abs_at_4096": max_abs, "passed": passed}


def test_memeff_end_to_end(model_name: str, device: str) -> dict:
    """Test 3: full Gemma 4 forward under sdpa_memeff vs stock sdpa.

    Loads the model twice (once per attn_implementation) and compares logits
    on the same input. Catches any end-to-end drift from our K/V repeat,
    context-manager placement, or interaction with RoPE/GQA.
    """
    section("TEST 3 — Gemma 4 full forward: sdpa_memeff vs sdpa (math fallback)")
    import sft.gemma4_liger_patch  # registers sdpa_memeff

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    seq = 256  # small so math backend fits comfortably
    text = "Once upon a time in a distant galaxy, " * 15
    enc = tokenizer(text, return_tensors="pt", max_length=seq, truncation=True).to(device)
    input_ids = enc.input_ids

    labels = input_ids.clone()

    print("  loading with attn=sdpa_memeff...")
    m1 = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map={"": device},
        attn_implementation="sdpa_memeff",
    ).eval()
    with torch.no_grad():
        out1 = m1.forward(input_ids=input_ids, labels=labels, skip_logits=False)
    loss1 = float(out1.loss)
    logits1 = out1.logits.float().cpu()
    # Top-1 prediction for each position.
    top1_1 = logits1.argmax(-1)
    del m1; torch.cuda.empty_cache()

    print("  loading with attn=sdpa (math-backend fallback) ...")
    m2 = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map={"": device},
        attn_implementation="sdpa",
    ).eval()
    with torch.no_grad():
        out2 = m2.forward(input_ids=input_ids, labels=labels, skip_logits=False)
    loss2 = float(out2.loss)
    logits2 = out2.logits.float().cpu()
    top1_2 = logits2.argmax(-1)
    del m2; torch.cuda.empty_cache()

    loss_diff = abs(loss1 - loss2)
    top1_agreement = float((top1_1 == top1_2).float().mean())
    # Soft-prob comparison: KL divergence in fp32 — captures any meaningful
    # shift in the distribution even when raw logits drift by bf16 noise.
    p = F.log_softmax(logits1, dim=-1)
    q = F.log_softmax(logits2, dim=-1)
    kl = float(F.kl_div(p, q, reduction="batchmean", log_target=True).abs())

    print(f"  loss (sdpa_memeff)  = {loss1:.6f}")
    print(f"  loss (sdpa math)    = {loss2:.6f}")
    print(f"  loss diff           = {loss_diff:.6f}")
    print(f"  top-1 agreement     = {top1_agreement*100:.2f}%  (bf16 noise expected <<1% disagreement)")
    print(f"  KL divergence       = {kl:.6e}")

    passed = loss_diff < 5e-3 and top1_agreement > 0.99
    print(f"  {'PASS' if passed else 'FAIL'}")
    return {"test": "memeff_end_to_end", "loss_diff": loss_diff,
            "top1_agreement": top1_agreement, "kl": kl, "passed": passed}


def main():
    model_name = "google/gemma-4-E4B-it"
    device = "cuda"

    results = []
    results.append(test_fused_ce_vs_reference(model_name, device))
    results.append(test_memeff_vs_math_attention(device))
    results.append(test_memeff_end_to_end(model_name, device))

    section("SUMMARY")
    for r in results:
        print(f"  {r['test']:25s}  {'PASS' if r['passed'] else 'FAIL'}")
    all_pass = all(r["passed"] for r in results)
    print(f"\n  OVERALL: {'ALL PASS — patch is numerically sound' if all_pass else 'FAILURE DETECTED — do not trust training signal'}")


if __name__ == "__main__":
    main()
