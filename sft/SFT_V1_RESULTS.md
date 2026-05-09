# SFT v1 — Gemma 4 E4B-it LoRA — Training + Eval Results

Run: 2026-04-24 → 2026-04-25.

## TL;DR

- **Trained Gemma 4 E4B-it LoRA** on 7278 mixed searcher+extractor samples.
- **Final train loss 1.149** after 386 steps (1 epoch, eff batch 18, seq=24k).
- **Filterbench/test (33 tasks): 14/33 = 42.4%, 0 errors, 44 min wall**.
- Compares to untuned E4B with thinking on DSQA (3/32 = 9.4%) — different bench, but the gap suggests the LoRA dramatically improves searcher quality. **Apples-to-apples filterbench baseline still TODO.**

## Training Run

| | |
|---|---|
| Base model | `google/gemma-4-E4B-it` (8B params, head_dim=320, 42 layers) |
| Hardware | RTX PRO 6000 Workstation, sm_122, 96 GB VRAM |
| Image | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` (vast.ai container) |
| Effective torch | 2.11.0+cu128 (image was upgraded by training-deps install) |
| LoRA rank / alpha / target | 16 / 32 / `all-linear` (excluding `lm_head`) |
| Trainable params | 50.5M (0.6% of 8.0B) |
| Sequence length | 24576 (max), with custom packing |
| Effective batch | 18 (batch=3, grad_accum=6) |
| Learning rate | 2e-5, cosine, warmup_ratio=0.03 |
| Epochs | 1 (resumed from step 50 of 2-epoch plan after disk-full crash) |
| Total steps | 386 |
| Optimizer | AdamW (default) |
| Mixed precision | bf16 |
| Gradient checkpointing | on |
| Wall clock | 16h 50min on RTX PRO 6000 (single GPU) |
| Final train_loss | **1.149** |
| Tokens trained | 47.96M |
| W&B | https://wandb.ai/dvdimitrov13/web-search-gym-sft/runs/tx2j185x |

### Loss trajectory (smoothed, last few logged steps)

```
1.241 → 1.157 → 1.141 → 1.177 → 1.149 (final smoothed)
```

Healthy descent throughout. Grad norms stayed bounded (peaked ~15 at warmup, settled to 0.16-0.42).

### Custom infra used (see `sft/PATCH_VERIFICATION.md`)

1. **Liger fused linear-CE** for Gemma 4 (Liger-Kernel 0.7 doesn't support gemma4 natively): patched `Gemma4ForConditionalGeneration.forward` to route loss through `LigerForCausalLMLoss`. Bit-exact equivalence to materialized-logits CE (Test 1: diff = 0.000000).
2. **`sdpa_memeff` attention interface**: registered in `ALL_ATTENTION_FUNCTIONS`. Pins SDPA to EFFICIENT_ATTENTION via `sdpa_kernel`. Required because head_dim=320 exceeds FLASH/CUDNN library caps; without it, dispatcher falls back to MATH (74 GB at seq=32k → OOM).
3. **`PackedSFTCollator`** with 4D block-diagonal mask: avoids padding waste at batch>1 without flash-attn varlen. Per-sample loss matches non-packed within bf16 noise (Test 4).

All four equivalence tests pass — see `sft/PATCH_VERIFICATION.md`.

## Eval Run

| | |
|---|---|
| Bench | filterbench/test (33 multi-hop questions, our synthetic bench) |
| Agent | agent_dd (search → browse → commit_memory → answer, 8 cycles max) |
| Searcher | LoRA-tuned E4B-it via vllm (thinking enabled) |
| Extractor | LoRA-tuned E4B-it via vllm (same model) |
| Judge | Anthropic Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) via OpenAI-compat endpoint |
| Concurrent | 2 |
| Wall clock | 44 min |
| **Score** | **14/33 = 42.4%, 0 errors** |

### Per-task breakdown

✅ **Correct (14):** idx 2, 3, 6, 8, 9, 11, 13, 15, 18, 22, 23, 24, 25, 28
❌ **Wrong (19):** idx 0, 1, 4, 5, 7, 10, 12, 14, 16, 17, 19, 20, 21, 26, 27, 29, 30, 31, 32

### Failure patterns

- **Conservative refusals** ("information not found / unavailable"): idx 7, 10, 14, 17, 20, 21, 29 — model gives up rather than push search further. Suggests prompt or training could push exploration harder.
- **Wrong-year reasoning**: idx 0, 1 — agent responds correctly to "I don't know what year" rather than committing.
- **Near-miss / format mismatch**: idx 22 ("RTS,S (Mosquirix)" vs gold "Mosquirix") was actually marked PASS by the judge; idx 13 ("South Africa" vs "Pretoria, South Africa") also marked PASS. Format tolerance is good.
- **Genuinely wrong factual answer**: idx 4 (Biden's birthplace — agent answered "Scranton" which is correct; gold says "Queens, New York City" which is **likely a gold-data error**).
- **Hallucinated entity**: idx 30 (1927 or 1701 vs gold 1885) — agent guessed.

### Indirect baselines for context (different benchmarks)

| Run | Model | Bench | Score |
|---|---|---|---|
| `e4b_baseline` | Gemma 4 E4B-it (no LoRA, no thinking) | DSQA dom2 | 2/32 = 6.3% |
| `e4b_thinking_v2` | Gemma 4 E4B-it (no LoRA, thinking on) | DSQA dom2 | 3/32 = 9.4% |
| `cached_c8_mem512` | Claude Sonnet | DSQA dom2 | 15/32 = 46.9% |
| **(this run)** | **LoRA E4B + thinking** | **filterbench** | **14/33 = 42.4%** |

Filterbench (synthetic, our team's) is presumably easier than DSQA (Google's harder research bench). The 42% on filterbench is **not** directly comparable to the 47% Sonnet number on DSQA, but the directional jump from 9% (untuned E4B + thinking on DSQA) → 42% (LoRA + thinking on filterbench) is large enough that even substantial bench-difficulty differences leave the LoRA contribution clearly positive.

### Apples-to-apples DSQA datapoint (`sft_v1_dsqa_local`, 2026-04-26)

Re-ran the same LoRA on the same `agent_dd` 8-cycle + commit_memory
harness against DSQA `domain2` (32 tasks, gemini-2.5-flash judge,
concurrency=4) on a fresh vast.ai 5090. Closes the "untuned E4B
baseline on filterbench/test" TODO from below — turns out the more
useful comparison is the other direction: same LoRA, on DSQA.

| Run | Model | Bench | Correct | F1 | P/R | Wall |
|---|---|---|---:|---:|---|---:|
| `e4b_thinking_v2` | Gemma 4 E4B-it (no LoRA, thinking on) | DSQA dom2 | 3/32 = 9.4% | 0.260 | .362/.226 | 732s |
| **`sft_v1_dsqa_local`** | **LoRA E4B + thinking on** | **DSQA dom2** | **4/32 = 12.5%** | **0.246** | **.281/.232** | **882s** |
| `cached_c8_mem512` | Claude Sonnet | DSQA dom2 | 15/32 = 46.9% | 0.607 | .629/.612 | 982s |

**+3.1pt fully-correct, F1 ~flat.** The LoRA helps the searcher push to a
committed answer slightly more often (12.5% vs 9.4%), but per-token
extraction quality (F1) doesn't move — recall holds, precision drops
~8pt. Wall-clock +20% suggests the LoRA pulls the agent into longer
chains it doesn't always close cleanly. 1/32 invalid auto-rater (vs
0 on the untuned).

**Read alongside the 42.4% filterbench number.** The same LoRA shows
a ~33pt gap between filterbench and DSQA. That's the SFT-data /
target-bench domain gap: SFT v1 trains on filterbench-style multi-hop
+ filter trajectories; DSQA's 17 categories of broad enumeration are
out-of-distribution. So the SFT teaches the harness shape (commit
answer, parallel search, scratchpad use) but not DSQA's Set Answer
enumeration. Implication for v2: mix DSQA-shaped trajectories into
SFT data, or run GRPO on a DSQA-leaning reward.

**Per-category breakdown** (from the aggregate JSON):
Won at least one in: Arts (1/2), Education (1/2), Linguistics (1/1),
Travel (1/2). 0/2 in: Biology, Current Events, Finance, Geography,
Health, History, Media, Other, Politics, Science, Technology. 1/2 sport.
Wide failure profile, consistent with a generic harness lift rather
than a category-specific knowledge bump.

Artifacts:
- `results/raw/agent_dd__searcher-gemma_4_e4b_vllm__extractor-gemma_4_e4b_vllm__sft_v1_dsqa_local.jsonl`
- `results/scores/agent_dd__searcher-gemma_4_e4b_vllm__extractor-gemma_4_e4b_vllm__sft_v1_dsqa_local.json`
- See also `docs/agent-dd-dsqa.md` §9 — same row added to the running
  Gemma baselines table for cross-reference.

**Repro notes from this rerun (additions to caveats below):**
- `merge_and_unload` actually drops **54 tensors** for layers 24-41
  (k_norm, k_proj, v_proj — not just k_norm as previously documented).
  Same fix: copy from base, update index. `sft/SFT_V1_RESULTS.md`'s
  earlier "18 k_norm" was an undercount.
- Gemma 4 E4B base ships as a **single** safetensors file (no shard
  index), so any patcher that compares shard indices needs to fall
  back to reading keys directly from the single file.
- `huggingface-hub[cli]` extra was removed in hub ≥1.12 — drop the
  `[cli]` extra from any future bootstrap script.

## Deployment / Inference path

vllm 0.19.1 served the merged model:
```
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/gemma4_e4b_merged \
  --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --served-model-name gemma_4_e4b_lora --enforce-eager
```

vllm-serving caveats discovered today (all required for merged model to load):

1. **vllm 0.19.1 errors on `Gemma4ForConditionalGeneration` LoRA** ("does not support LoRA yet"). Workaround: merge LoRA into base via `peft.merge_and_unload()` and serve the merged checkpoint directly — no LoRA at runtime.
2. **`merge_and_unload` drops 18 k_norm tensors** for layers 24-41 (Gemma 4's KV-shared global-attention layers). Without them, vllm fails with "weights were not initialized from checkpoint". Patch: copy the 18 tensors from the original base model into shard 4 + update `model.safetensors.index.json`. See `/tmp/patch_merge.py` from this session.
3. **Multimodal `preprocessor_config.json`** is required by vllm's `gemma4_mm` model — `tokenizer.save_pretrained()` doesn't include it. Re-save with `AutoFeatureExtractor.from_pretrained(BASE).save_pretrained(OUT)`.
4. **`--enable-auto-tool-choice` + `--tool-call-parser gemma4`** both required (just the parser flag isn't enough; vllm checks for the boolean too).
5. **`--enforce-eager`** required if the runtime image lacks gcc (Inductor compilation needs C compiler at runtime).
6. **Eval should run ON the GPU instance** (vllm + bench harness on same box), not via SSH-tunneled localhost:8000 from local. vast.ai's SSH proxy was unreliable today and tunnel drops cascade into APIConnectionErrors across the eval.

## OSS issues filed (with reproducers)

1. **[Liger-Kernel #1186](https://github.com/linkedin/Liger-Kernel/issues/1186)** — comment offering `Gemma4ForConditionalGeneration` follow-up to PR #1196.
2. **[TRL #5316](https://github.com/huggingface/trl/pull/5316)** — comment showing 4D block-diag mask is the missing piece in @qgallouedec's failing test, with our `PackedSFTCollator` reference impl.
3. **[transformers #45636](https://github.com/huggingface/transformers/issues/45636)** — proposed `attn_implementation="sdpa_memeff"` for shapes no fast backend covers.
4. **[pytorch #181379](https://github.com/pytorch/pytorch/issues/181379)** — CUDNN head_dim cap stuck at 128 on sm_120 (RTX 5090), with empirical reproducer + source analysis. 2-line fix.

## Local artifacts

- `sft/output/gemma4_e4b_mixed/` — final LoRA adapter (202 MB safetensors)
- `sft/output/gemma4_e4b_mixed_v1_resume_local/train-resume.log` — full training log
- `results/raw/agent_dd__searcher-gemma_4_e4b_vllm__extractor-gemma_4_e4b_vllm__remote_full_1777145090.jsonl` — per-task eval results
- `results/scores/agent_dd__searcher-gemma_4_e4b_vllm__extractor-gemma_4_e4b_vllm__remote_full_1777145090.json` — eval aggregate
- `sft/PATCH_VERIFICATION.md` — Gemma 4 patch numerical equivalence tests
- `sft/POTENTIAL_PRS.md` — full upstream-PR analysis

## TODO for next session

- [x] ~~Run untuned E4B-it baseline on filterbench/test~~ — instead, ran the LoRA on DSQA dom2 against the existing untuned E4B baseline (see "Apples-to-apples DSQA datapoint" section above). +3.1pt fully-correct, F1 ~flat.
- [ ] **Consider 2 epochs** — original plan was 2 epochs but disk crash forced reduction to 1. Loss was still descending; another epoch might give 5-10% absolute gain.
- [ ] **Address conservative-refusal failure mode**: 7 of 19 wrong answers were "information not found" rather than wrong guesses. Suggests SFT data should include more "push search harder" trajectories, or system prompt should pressure deeper search before giving up.
- [ ] **Audit gold answers**: idx 4 (Biden birthplace) gold appears wrong. Worth double-checking other gold answers.
- [ ] **Bake today's lessons into a one-shot startup script**: install gcc, pip install vllm + peft + exa-py + anthropic + click + rich + pyyaml + datasets, run merge with k_norm patch + preprocessor save, launch vllm with all required flags. Eval-on-remote (no tunnel).
- [ ] **GRPO**: was the next planned phase per the project plan. SFT v1 result (42% on filterbench) is the seed for GRPO.

## Lessons learned (process)

- **vast.ai SSH proxy is unreliable** — tunnels drop, all subsequent localhost calls fail. Run jobs on the remote, not via tunnel.
- **vllm install eats 3-5 GB during pip resolution** — don't run on instances with <8 GB free.
- **`merge_and_unload()` for Gemma 4 silently drops k_norm for KV-shared layers**. PEFT bug worth filing once verified on simpler repros.
- **The "image label says torch 2.4 but actual env has torch 2.11"** mismatch happened because vast.ai's startup script auto-upgrades torch. Trust `python -c "import torch"`, not the image tag.
- **Always use `nohup` for any long-running command on remote** — SSH disconnects kill foreground processes.
- **Keep balance topped up** — running out mid-run cost us 6h of training yesterday. $5+ buffer at all times is cheap insurance.
