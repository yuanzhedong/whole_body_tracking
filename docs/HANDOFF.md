# Handoff — running the G1 motion-tokenizer pipeline

A shareable guide for someone else to clone and run this repo.

**What it is:** A public fork of the BeyondMimic motion-tracking codebase
(`HybridRobotics/whole_body_tracking`) that adds a **seed-data → UniMoTok VAE → BFM-Zero →
sim2sim** pipeline for validating that VAE-decoded G1 robot motion is physically executable in
MuJoCo.

- **Repo:** https://github.com/yuanzhedong/whole_body_tracking (public)
- **Branch with the pipeline + docs:** `seed-vae-sim2sim-pipeline` *(not yet merged to `main`)*

## Clone

```bash
git clone -b seed-vae-sim2sim-pipeline git@github.com:yuanzhedong/whole_body_tracking.git
cd whole_body_tracking
git submodule update --init --recursive   # see Blocker 1 below
```

**Read next:** [`PIPELINE_SETUP.md`](PIPELINE_SETUP.md) (full setup + exact commands per stage) and
[`README.md`](README.md) (docs index / diagram).

## The pipeline

```
BONES-SEED G1 CSV → 41-D features → UniMoTok MLD-VAE → decode → qpos_36 → BFM-Zero / HoloMotion tracker → MuJoCo
  stage2/seed_to_    stage2/export_    UniMoTok           stage3_sim2sim/    stage3_sim2sim/             survival /
  artifacts.py       g1_motion.py      (submodule)        decode_to_qpos36   run_l3_eval.py             tracking error
```

It answers: **is VAE-decoded G1 motion physically executable?** Stages 1–3 are self-contained in
this repo; Stage 4 (the physics validator) shells out to external repos.

**W&B result reports:**
- [Headline — BONES-SEED → UniMoTok VAE → BFM-Zero sim2sim](https://wandb.ai/toddler_tracking/g1-sim2sim/reports/x--VmlldzoxNzMzOTg2Mg==)
- [HoloMotion-validated pipeline (sibling)](https://wandb.ai/toddler_tracking/g1-sim2sim/reports/x--VmlldzoxNzMwNzgxMw==)
- [Tracker comparison — HoloMotion vs BFM-Zero](https://wandb.ai/toddler_tracking/g1-sim2sim/reports/x--VmlldzoxNzMzNDI0MA==)

## What you must supply yourself (not in the repo)

| # | blocker | why | how to resolve |
|---|---|---|---|
| 1 | **UniMoTok submodule access** | the submodule is `git@github.com:Juzezhang/UniMoTok.git` on branch `wbt-integration` (the pin `92ca9af` is pushed and carries the G1 training configs) — but it's an SSH remote that requires read access | request read access to `Juzezhang/UniMoTok`, then `git submodule update --init --recursive` resolves cleanly. (If you only have HTTPS, set the submodule URL accordingly.) |
| 2 | **OMG repo + HoloMotion ONNX** (Stage 4) | `stage3_sim2sim/run_l3_eval.py` hardcodes `/ws/user/yzdong/src/github/OMG` and the ONNX at `/scratch/user/yzdong/OMG-models/holomotion_dl/...` | obtain the OMG repo + ONNX; pass `--omg-root <path>` and edit the `ONNX` constant in `run_l3_eval.py` |
| 3 | **BFM-Zero repo** (`humanoidverse`) | imported by `stage3_sim2sim/bfmzero_compare/batch_tracking_inference.py`; path `/ws/.../BFM-Zero` | install BFM-Zero / `humanoidverse` in the Stage-4 env |
| 4 | **Seed CSVs** | `stage2/seed_to_artifacts.py --src` defaults to `/scratch/.../bones_seed/g1/csv` | obtain the BONES-SEED G1 CSVs (or any G1 CSVs in that schema) and pass `--src` |
| 5 | **Robot description assets** | the base repo pulls these from GCS | follow the base README install step (`curl ... unitree_description.tar.gz`) |
| 6 | **Trained VAE checkpoint** | reports use `g1_seed_512_fixed_FINAL.ckpt`; `.ckpt`/LFS weights are not committed | obtain the checkpoint, or retrain it (Stage 2) |

## Environment

- Isaac Sim 4.5 + Isaac Lab 2.1, Python 3.10. Install per the base
  [README](../README.md#installation): `python -m pip install -e source/whole_body_tracking`, plus
  `UniMoTok/requirements.txt` for the VAE env.
- The repo's local `.venv*/` dirs do **not** transfer — recreate them.
- GPU: validated on RTX 4090s. Blackwell GPUs are unusable with torch 2.5.1 (no kernel image).
- Required env vars for any Isaac step:
  `OMNI_KIT_ACCEPT_EULA=YES CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<4090> WANDB_ENTITY=<org>`.

## What runs without the external deps

Stages 1–3 (seed → features → VAE train/decode → BFM-Zero motion `.pkl`) run with the repo + the
UniMoTok submodule (blocker 1) + seed data (blocker 4). Stage 4 (physics validation) additionally
needs blockers 2–3.

See [`PIPELINE_SETUP.md`](PIPELINE_SETUP.md) for the exact command for each stage and the
verification gates.
