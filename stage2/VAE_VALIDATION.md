# G1-VAE Validation Log

Validation of the G1 motion VAE (UniMoTok `MldVaeBiomechanics`, 22.4M params) before
integration into OmniMM as a robot-motion modality (Option A: separate VAE per modality,
projected into OmniMM's shared LLM embedding space via a linear layer).

---

## Architecture: what the VAE does

```
reference G1 motion clip [T, 41]   (41-D: 6D-rot + lin_vel + ang_vel + 29 joint angles)
        │
   MldVaeBiomechanics.encode()
        │
   latent z  [1, 256]              (1 global token, 256-D — compresses full window)
        │
   MldVaeBiomechanics.decode()
        │
reconstructed motion [T, 41]
```

The VAE is **not** a robot controller — it encodes/decodes motion trajectories.
Downstream execution requires a separate Stage-0 BeyondMimic tracking policy.

---

## Validation pipeline (`stage2/sim2sim_vae_eval.py`)

```
reference motion clip
        │
        ├─── Stage-0 policy (original) ──► Isaac Sim ──► survival, E_mpbpe  [BASELINE]
        │
        ├─── G1-VAE encode → z → decode ──► reconstructed motion
        │         │
        │    Phase 0: offline RMSE          (joint angle, root orient, root vel errors)
        │         │
        │    Phase 1: Isaac kinematic FK    (decoded joint angles → full body state npz)
        │         │
        └─── Phase 1 output → Stage-0 policy ──► Isaac Sim ──► survival, E_mpbpe  [DECODED]
                                                                        │
                                                               Phase 2 comparison:
                                                         decoded/original ratio → PASS/FAIL
```

**Pass criteria:**
| phase | metric | threshold |
|---|---|---|
| Phase 0 | joint angle RMSE | < 0.10 rad (OmniMM biomech benchmark) |
| Phase 0 | root orient error | < 10° geodesic |
| Phase 2 | survival ratio (decoded / original) | ≥ 0.90 |
| Phase 2 | E_mpbpe ratio (decoded / original) | ≤ 1.15 |

---

## Run commands

```bash
# Phase 0+1 only (no Isaac, fast — VAE encode/decode + decoded npz creation):
.venv/bin/python stage2/sim2sim_vae_eval.py \
    --vae_ckpt UniMoTok/experiments/biomechanics_tokenizer/G1_MldVAE_v1/checkpoints/epoch=564.ckpt \
    --dataset_dir stage2/out/g1_dataset_yup \
    --teacher_ckpt logs/rsl_rl/g1_flat/2026-05-30_15-59-30_walk_4090_full30k/model_29999.pt \
    --splits val test --out stage2/out/sim2sim_eval.json --phase01_only

# Full pipeline (Phase 0+1+2, requires Isaac — only run when NO other Isaac process is running):
# NOTE: Phase 2 blocked while any other Isaac process (e.g. run policy training) is running.
# The DriverShaderCacheManager is machine-wide; only the first Isaac process to start can init it.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=YES \
  .venv/bin/python stage2/sim2sim_vae_eval.py \
    --vae_ckpt UniMoTok/experiments/biomechanics_tokenizer/<ckpt>/epoch=N.ckpt \
    --dataset_dir stage2/out/g1_dataset_yup \
    --teacher_ckpt logs/rsl_rl/g1_flat/.../model_29999.pt \
    --splits val test --out stage2/out/sim2sim_eval.json
```

**IMPORTANT — Isaac concurrency constraint:** Only one Isaac Sim process can hold the
`DriverShaderCacheManager` at a time. This is a machine-wide singleton (not per-GPU).
All Isaac jobs (training, eval, verify, sim2sim Phase 2) must run serially. The first
process to start initializes it; subsequent ones get a "init without shutdown" warning
and have a broken PhysX CUDA pipeline → "no kernel image" errors in gym.make.

---

## Results log

### Run 1 — 2026-06-05 — G1-VAE v1, epoch 564 (best val checkpoint)

**Training details:**
- Data: 12 LAFAN1/walk clips, 36,795 train windows (y-up, 20 fps, 128-frame windows)
- Architecture: `MldVaeBiomechanics`, nfeats=41, latent=[1,256], 9-layer encoder-decoder
- Best val loss: **3.481** at epoch 564 (bottomed out; overfitting from ~epoch 600 onward)
- WandB: https://wandb.ai/cs224n-robustqa/g1-motion-tokenizer/runs/z6ykfmkx

**Phase 0 — offline reconstruction:**

| clip | split | joint angle RMSE | root orient error | root lin vel MAE | PASS |
|---|---|---|---|---|---|
| lafan_fallAndGetUp1_subject1 | val  | **0.418 rad** | 73.9° | 0.336 m/s | ❌ |
| dance1_subject2              | test | **0.359 rad** | 31.7° | 0.212 m/s | ❌ |

**Verdict: FAIL.** Joint angle RMSE is 4–5× above the 0.10 rad target. Root orientation error
is very large (31–74°). The VAE is not reconstructing motion faithfully enough.

**Phase 1 (2026-06-06, smoke test) — PASS:**
- Pipeline: encode clip → decode → splice decoded joint angles into original motion npz (root from original).
- `lafan_fallAndGetUp1_subject1_decoded.npz` created successfully: 8410 frames, 29 joints replaced.
- Decoded npz format valid — all required keys present (`joint_pos`, `body_pos_w`, etc.).

**Phase 2 — BLOCKED** (Isaac `DriverShaderCacheManager` held by run policy):
- `gym.make` fails with "no kernel image" when any other Isaac process is running.
- Will run when run policy (lafan_run1_subject2) finishes (~6h from 2026-06-06 17:39).

**Root cause — data starvation:**
- 12 clips = 581 train windows. A 22.4M-param Transformer VAE needs far more data.
- Val loss bottomed at epoch ~600 and diverged: **train=0.52, val=4.22 at epoch 5401**
  (train–val gap of 3.7 units = severe overfitting).
- The two eval clips (`lafan_fallAndGetUp1_subject1`, `dance1_subject2`) are out-of-distribution
  from the train set (different motion categories than the walk-heavy training data).

**What's needed to pass:**
1. **More training data** — need run, sprint, jump, dance, fight tracking policies trained
   (queued; launching on GPU 1 after quality eval finishes on GPU 5).
2. **Larger, category-balanced train/val/test split** — eval clips should be held-out from
   categories the model also trains on, not purely OOD.
3. **Re-export with quality filter** — `stage2/out/track_quality.json` (in progress) will
   filter low-quality clips and ensure only well-tracked motions enter training.
4. **Re-train G1-VAE v2** on the expanded dataset → target joint RMSE < 0.10 rad.

---

## Next run (planned: G1-VAE v2)

Trigger: quality eval done + ≥4 new tracking policies (run/sprint/jump/dance) complete.

Expected dataset: ~50–80 clips, ~150k+ train windows.

Expected Phase 0 targets:
- Joint angle RMSE < 0.10 rad
- Root orient error < 10°

Expected Phase 2 targets:
- Survival ratio ≥ 0.90
- E_mpbpe ratio ≤ 1.15

When Phase 2 passes → VAE is validated → hand G1-VAE checkpoint + normalization stats
to Yao He for OmniMM branch wiring (linear projector into shared LLM space).
