# Pipeline verification — end-to-end on RTX PRO 6000 Blackwell

Run **2026-06-28** on this machine (4× RTX PRO 6000 Blackwell sm_120 + 2× RTX 4090), env `.venv6`
(Python 3.12, torch `2.10.0+cu128`). Reproduces the seed → UniMoTok VAE → decode → HoloMotion →
MuJoCo chain end-to-end. All stages pass.

## Environment sanity (Blackwell, sm_120)

| check | env | result |
|---|---|---|
| torch CUDA matmul on RTX PRO 6000 (cap 12.0) | `.venv6` (torch 2.10/cu128) | ✅ finite |
| same on torch 2.5.1+cu124 (`.venv`, Isaac path) | `.venv` | ❌ "no kernel image" (sm ≤ 90) → confirms the 4090-only limit is **torch-version**, not hardware |

## Stage-by-stage

| stage | what was run | result |
|---|---|---|
| **1** seed→features / inverse | real artifact `motion.npz` → `qpos36_to_features` → `features_to_qpos36` | joint round-trip MAE **0.0000°** (exact passthrough) |
| **2/3** VAE load+decode | load 61M-param `g1_seed_512_fixed_FINAL.ckpt`, encode→decode a real val clip on Blackwell | finite output, joint recon RMSE **5.83°** (single clip) |
| physics engine | MuJoCo 3.5.0 build + 10 steps | ✅ qpos finite |
| **4** full e2e (`run_l3_eval.py`) | VAE decode (Blackwell) → `qpos_36` → HoloMotion ONNX v1.3.1 tracker (OMG `.venv-cu128`) → MuJoCo | see Gate D below |

## Gate D — decoded vs original (2 walk clips, hybrid root)

| clip | decode joint RMSE | survival orig | survival decoded | track orig | track decoded |
|---|---|---|---|---|---|
| amass_ACCAD_Male2Walking | 9.20° | 1.00 | 1.00 | 20.4° | 20.7° |
| amass_accad_qkwalk1 | 11.51° | 1.00 | 1.00 | 21.1° | 18.0° |
| **mean** | **10.36°** | **1.000** | **1.000** | — | — |

**Decoded motion survives in MuJoCo exactly like the original (1.00 vs 1.00)** and tracks within
the original's error band — the VAE-reconstructed joint motion is physically executable. Matches the
SIM2SIM_PLAN Gate-D claim (decoded ≈ original survival).

## How to reproduce

```bash
# Stages 1-3 + the env sanity run in .venv6 (Blackwell). Stage 4 shells out to
# OMG's own .venv-cu128 tracker env + the HoloMotion ONNX (external; see HANDOFF.md).
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<blackwell-idx> \
PYTHONPATH=.:UniMoTok .venv6/bin/python stage3_sim2sim/run_l3_eval.py \
    --cfg  UniMoTok/configs/config_g1_seed_512_fixed.yaml \
    --ckpt UniMoTok/experiments/_compare/g1_seed_512_fixed_FINAL.ckpt \
    --umt-root UniMoTok --omg-root /ws/user/yzdong/src/github/OMG \
    --clips artifacts/<clipA>/motion.npz artifacts/<clipB>/motion.npz \
    --out /tmp/l3_out --num-frames 128 --device cuda
```

## Caveats

- Stage 4 needs the **external** OMG repo + HoloMotion ONNX (`run_l3_eval.py` `ONNX` const) — present
  on this machine, not in the repo. The BFM-Zero validator path is the sibling under
  `stage3_sim2sim/bfmzero_compare/` (needs the BFM-Zero `humanoidverse` env).
- The VAE-inference deps (`omegaconf`, `pytorch-lightning`, `pandas`, `loguru`, `matplotlib`) were
  added to `.venv6` during this verification — now pinned in
  [`requirements-venv6-core.txt`](requirements-venv6-core.txt) and the full freeze.
- 2-clip Gate D is a smoke of the full chain on Blackwell; the large-scale survival numbers live in
  the W&B reports (see [`PIPELINE_SETUP.md`](PIPELINE_SETUP.md#7-results--wb-reports)).
