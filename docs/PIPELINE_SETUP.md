# Seed → UniMoTok VAE → BFM-Zero → sim2sim — pipeline setup

End-to-end guide to reproduce the motion-tokenizer pipeline in this repo:

```
BONES-SEED G1 motion ──► 41-D features ──► UniMoTok MLD-VAE ──► decode ──► qpos_36 ──► BFM-Zero / HoloMotion tracker ──► MuJoCo
   (CSV, 120 fps)         (heading-canon)    (encode→latent)    (latent→41-D)  (root+joints)   (RL generalist policy)        (does G1 stay
                                                                                                                              upright & on-trace?)
```

The pipeline answers one question: **is VAE-decoded G1 motion physically executable?** We learn a
UniMoTok motion latent on BONES-SEED G1 data, decode reconstructions, and run them through a
physics tracker (BFM-Zero and/or HoloMotion) in MuJoCo to measure survival and tracking error
against the original reference.

> **Results / W&B reports** are linked at the bottom. The headline: VAE-decoded motion stays
> executable (survival_rel ≈ 0.98 decoded vs original), and BFM-Zero extends physics coverage to
> near-ground crouch/squat/sit motion that HoloMotion collapses on.

---

## 0. What is and isn't self-contained

**Self-contained in this repo** (Stages 1–3 — the contributions): seed ingestion, the 41-D
feature build, the UniMoTok VAE (git submodule), decode, and the qpos_36 / BFM-Zero motion
conversion. These run with the repo's own venvs.

**External prerequisites** (Stage 4 — the physics validator only):

| dependency | role | how it's referenced |
|---|---|---|
| **OMG** repo | HoloMotion `tracker-only` MuJoCo driver | `run_l3_eval.py --omg-root` (default `/ws/user/yzdong/src/github/OMG`) |
| **HoloMotion ONNX** | generalist G1 tracker weights | `run_l3_eval.py` `ONNX` const (`/scratch/user/yzdong/OMG-models/holomotion_dl/...`) |
| **BFM-Zero** repo (`humanoidverse`) | Forward-Backward foundation-model tracker | imported by `stage3_sim2sim/bfmzero_compare/batch_tracking_inference.py` |

The validator is *intentionally* external: the tracker + MuJoCo physics were validated in the OMG
reproduction and are reused as-is. Point the `--omg-root` / ONNX path / `humanoidverse` install at
your local clones. Everything upstream of qpos_36 needs none of these.

---

## 1. Environment

This repo carries multiple Python venvs. Different stages use different ones:

| venv | torch | used for | Blackwell (sm_120)? |
|---|---|---|---|
| `.venv6/` | **2.10.0+cu128** | the seed→VAE→BFM-Zero→**sim2sim** pipeline (Stages 1–4; Isaac-free: numpy/scipy + torch + MuJoCo) | ✅ works |
| `.venv/` | 2.5.1+cu124 | Isaac Sim 4.5 + Isaac Lab 2.1 — *optional* RL teacher training, `csv_to_npz`, Isaac evals | ❌ sm ≤ 90 only |
| OMG / BFM-Zero env | (external) | Stage-4 tracker rollouts (lives with the external repo) | depends on its torch |

To rebuild the Blackwell-capable env, see **[`requirements-venv6-core.txt`](requirements-venv6-core.txt)**
(focused core deps + the cu128 install recipe) or **[`requirements-venv6.txt`](requirements-venv6.txt)**
(the full captured freeze, incl. Isaac Sim 6.0). Both pin torch `2.10.0+cu128` (Python 3.12).

**GPU note (verified):** the pipeline runs **end-to-end on RTX PRO 6000 Blackwell (sm_120)** using
the torch-2.10/cu128 env (`.venv6`). The earlier "Blackwell unusable" caveat is a **torch-version**
constraint, not a hardware one — it applies only to the Isaac Sim 4.5 venv (torch 2.5.1+cu124),
whose CUDA kernels stop at sm_90, so that *optional* teacher-training path needs an RTX 4090
(sm 8.9) instead.

Required env vars for any Isaac step:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1   # for the Isaac/torch-2.5.1 path use a 4090 (sm 8.9); torch 2.10 (.venv6) runs on Blackwell too
export WANDB_ENTITY=<your-org>  # NOT your personal username
```

UniMoTok is a git submodule pinned to a feature branch (the `main` branch is incomplete):

```bash
git submodule update --init --recursive
# UniMoTok must be on feat/biomechanics_tokenization (or wbt-integration), never main
```

---

## 2. Stage 1 — seed data → 41-D features

### 2a. BONES-SEED CSV → artifacts

`stage2/seed_to_artifacts.py` converts BONES-SEED G1 CSVs (36 cols, 120 fps; root translate/rotate
+ 29 joints in OMG order) into `artifacts/<name>/motion.npz` — the same schema
`export_g1_motion.py` consumes. It resamples 120→30 fps, builds the z-up root pose, and reorders
joints OMG→UniMoTok.

```bash
.venv/bin/python stage2/seed_to_artifacts.py \
    --src /scratch/user/yzdong/OMG-Data/raw/bones_seed/g1/csv \
    --out artifacts \
    --limit 0          # 0 = all clips
```

### 2b. artifacts → 41-D feature dataset

`stage2/export_g1_motion.py` builds the UniMoTok training dataset: per-clip 41-D velocity
representation = `[root_rot6d(6), root_lin_vel_local(3), root_ang_vel_local(3), joints(29)]`,
heading-canonicalized, with FK ground truth preserved for eval. Clip-level train/val/test split
(no window leakage).

```bash
.venv/bin/python stage2/export_g1_motion.py \
    --artifacts_dir artifacts \
    --out_dir stage2/out/g1_dataset_yup \
    --target_fps 20 --window 128 \
    --to_yup            # Isaac z-up → UniMoTok y-up basis change
```

Output: `stage2/out/g1_dataset_yup/{train,val,test}/*.npz` + `manifest.json` + `normalization.npz`.
Add `--quality_json <track_quality.json>` to drop low-survival clips, or `--verify_only` to
range-check before writing.

---

## 3. Stage 2 — train the UniMoTok MLD-VAE

The VAE is `MldVaeBiomechanics` (MLD SkipTransformer; compresses a 128-frame window → a few global
latent tokens, then reconstructs). It is generic over `nfeats` — here `vae_test_dim: 41`.

Generate a config variant and train (run from `UniMoTok/`):

```bash
# (optional) make a config variant pointing at the dataset + an experiment dir
.venv/bin/python stage2/gen_vae_config.py --name g1_seed_512 \
    --data_dir $(pwd)/stage2/out/g1_dataset_yup \
    --exp_dir  $(pwd)/UniMoTok/experiments/biomechanics_tokenizer/g1_seed_512 \
    --out UniMoTok/configs/config_g1_seed_512.yaml \
    --latent 512

cd UniMoTok
python -m training.train_tokenizer --cfg configs/config_g1_seed_512.yaml --nodebug
```

Checkpoints land in `experiments/biomechanics_tokenizer/<name>/checkpoints/`. Training is logged to
W&B project `cs224n-robustqa/g1-motion-tokenizer`. The pipeline reports use the
`g1_seed_512_fixed_FINAL.ckpt` checkpoint.

**Verify reconstruction** (RMSE on held-out clips, no Isaac):

```bash
.venv/bin/python stage2/eval_vae_rmse.py --ckpt <ckpt> --cfg <cfg> \
    --data_dir stage2/out/g1_dataset_yup/test
```

> The latent is **not** N(0,I) out of the box — for any downstream generative use ship the
> per-dim standardization stats alongside the checkpoint (see [memory: OmniMM handoff]).

---

## 4. Stage 3 — decode → qpos_36 → BFM-Zero motion

`stage3_sim2sim/vae_decode_clip.py` (`decode_features`) runs encode→decode and returns the
reconstructed 41-D features. `stage3_sim2sim/decode_to_qpos36.py` inverts the 41-D map back to
`qpos_36 = [root_pos(3), root_quat_wxyz(4), joints(29)]` (see the inverse derivation in
`stage3_sim2sim/SIM2SIM_PLAN.md`).

> **Known limitation (handled):** the upstream double-yup convention corrupts the *absolute world
> root height* (joints + tilt invert exactly, world translation does not). sim2sim therefore uses a
> **hybrid reference** = decoded joints + the clip's original root, isolating "are the decoded
> joints executable?" from a root-reconstruction artifact. See `build_hybrid_qpos36` in
> `stage3_sim2sim/sim2sim.py`.

Convert decoded clips to a BFM-Zero motion `.pkl` (qpos_36 in OMG joint order → axis-angle
`pose_aa`):

```bash
.venv/bin/python stage3_sim2sim/to_bfmzero_motion.py \
    --artifacts artifacts/<clipA> artifacts/<clipB> \
    --out /path/to/BFM-Zero/pretrained/data/seed_decoded.pkl \
    --fps 30
```

---

## 5. Stage 4 — sim2sim physics validation

### Option A — HoloMotion tracker (OMG driver)

`stage3_sim2sim/run_l3_eval.py` is the end-to-end L3 driver: decode → qpos_36 → HoloMotion ONNX
tracker → MuJoCo rollout → survival / g_mpjpe / mpjpe / e_vel / e_acc. Needs the OMG repo + the
HoloMotion ONNX (edit the `ONNX` constant / `--omg-root` for your paths).

```bash
.venv/bin/python stage3_sim2sim/run_l3_eval.py \
    --cfg  UniMoTok/configs/config_g1_seed_512.yaml \
    --ckpt <g1_seed_512_fixed_FINAL.ckpt> \
    --umt-root UniMoTok \
    --omg-root /ws/user/yzdong/src/github/OMG \
    --clips artifacts/<clipA>/motion.npz artifacts/<clipB>/motion.npz \
    --out /tmp/sim2sim_l3 --num-frames 128 --device cpu
```

### Option B — BFM-Zero tracker (extends near-ground coverage)

Run the decoded `.pkl` through BFM-Zero's `humanoidverse` tracker. The batch runner
(`stage3_sim2sim/bfmzero_compare/batch_tracking_inference.py`) loads a BFM-Zero checkpoint and
rolls each clip out in MuJoCo; `decode_to_bfm.py` builds the decoded `.pkl` first. Run inside the
BFM-Zero / OMG env (it imports `humanoidverse`).

```bash
# in the BFM-Zero env, from this repo root
python stage3_sim2sim/bfmzero_compare/decode_to_bfm.py        # decoded clips → quant_decoded.pkl
python stage3_sim2sim/bfmzero_compare/batch_tracking_inference.py \
    --model-folder <bfm_zero_ckpt_dir> --data-path <decoded.pkl> --out-dir <out>
```

BFM-Zero matters because it **executes near-ground crouch / sit / squat** that HoloMotion collapses
on — so a BFM-Zero-validated pipeline certifies more of the motion distribution.

### Metrics

`rollout_metrics` (pure-numpy, unit-tested) reports **survival**, **survival_rel** (reference-relative —
the executed pelvis may sit `REL_MARGIN` below the reference pelvis before counting as fallen),
joint error (deg), and velocity/acceleration error. `survival_rel` is the headline executability
metric.

---

## 6. Verification gates (stop-and-verify)

The pipeline is gated end-to-end (`SIM2SIM_PLAN.md`); do not proceed past a red gate:

| gate | proves | status |
|---|---|---|
| **A** (L0 unit) | rotation/decode round-trip; synthetic `qpos→features→invert→recover` | PASS (joints exact, orient ≤0.32°, root drift ≤1.1 mm) |
| **L1** pairwise | real-clip round-trip (joint MAE; report root drift); VAE↔inverse | PASS (joints exact; world-root height is the documented xfail) |
| **B** validate-the-validator | original motion survives in MuJoCo | PASS (survival 1.00) |
| **C** decoded survives | decoded motion executes | PASS (survival 1.00, tracks 18.9° vs 19.7°) |
| **D** decoded-vs-original | survival/mpjpe gap across motion types | PASS (decoded 0.969 vs original 0.961) |

Run the unit/round-trip tests:

```bash
.venv/bin/python -m pytest stage3_sim2sim/tests -q
```

---

## 7. Results & W&B reports

All three live in the `toddler_tracking/g1-sim2sim` project. Regenerate with the
`create_*_report.py` scripts in `stage3_sim2sim/bfmzero_compare/` (they use `report.get_share_url()`
— never hand-construct report URLs).

- **G1 motion pipeline — BONES-SEED → UniMoTok VAE → BFM-Zero sim2sim** (the headline pipeline report):
  https://wandb.ai/toddler_tracking/g1-sim2sim/reports/x--VmlldzoxNzMzOTg2Mg==
  - Result 1: VAE-decoded motion stays executable in BFM-Zero (survival_rel original → decoded).
  - Result 2: BFM-Zero extends physics coverage to near-ground crouch/squat/sit.
- **HoloMotion-validated pipeline** (sibling; same pipeline, HoloMotion as the validator):
  https://wandb.ai/toddler_tracking/g1-sim2sim/reports/x--VmlldzoxNzMwNzgxMw==
- **Tracker comparison — HoloMotion vs BFM-Zero** (why the validator choice matters):
  https://wandb.ai/toddler_tracking/g1-sim2sim/reports/x--VmlldzoxNzMzNDI0MA==

Related projects/registry:
- `cs224n-robustqa/g1-motion-tokenizer` — UniMoTok MLD-VAE training runs.
- `cs224n-robustqa/g1-vae-ablation` — VAE data-scaling ablation.
- `cs224n-robustqa/wandb-registry-Motions` — motion.npz artifacts (LAFAN1 + AMASS + seed).

---

## 8. File map (where each stage lives)

| stage | entry point | output |
|---|---|---|
| 1a seed → artifacts | `stage2/seed_to_artifacts.py` | `artifacts/<name>/motion.npz` |
| 1b artifacts → features | `stage2/export_g1_motion.py` | `stage2/out/g1_dataset_yup/` |
| 2 VAE train | `UniMoTok` `training.train_tokenizer` (+ `stage2/gen_vae_config.py`) | `*.ckpt` |
| 2 VAE eval | `stage2/eval_vae_rmse.py` | RMSE json |
| 3 decode | `stage3_sim2sim/vae_decode_clip.py`, `decode_to_qpos36.py` | qpos_36 |
| 3 → BFM-Zero motion | `stage3_sim2sim/to_bfmzero_motion.py` | `*.pkl` |
| 4 sim2sim (HoloMotion) | `stage3_sim2sim/run_l3_eval.py` (+ `sim2sim.py`) | survival/tracking json |
| 4 sim2sim (BFM-Zero) | `stage3_sim2sim/bfmzero_compare/{decode_to_bfm,batch_tracking_inference}.py` | rollout npz |
| reports | `stage3_sim2sim/bfmzero_compare/create_*_report.py` | W&B reports |

For deeper rationale see `stage3_sim2sim/SIM2SIM_PLAN.md` (inverse derivation, gates) and
`stage2/g1_omnimm_modality_spec.md` (41-D representation, OmniMM hand-off).
