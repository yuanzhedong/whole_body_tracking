# BONES-SEED → Humanoid Data Generation Pipeline

Reproducible pipeline that turns the **BONES-SEED** motion dataset into two paired, training-ready datasets:

1. **Robot state–action** — the Unitree **G1** tracking the ground-truth retarget under the **BFM-Zero** policy, logged as `(reference motion, robot state) → action` tuples.
2. **Human SMPL / SMPL-X** — the same motions as parametric human body params (foot-corrected).

Both are keyed by the same clip name, so every robot clip has a matching human clip.

> **Design rules baked in (important):**
> - **No VAE anywhere.** The robot tracks the *ground-truth* retarget (a partially-trained motion VAE was tried and discarded — it destroyed wrist/waist motion). BFM-Zero policy inference is used (that's the source of the action labels); no other model inference.
> - **SMPL-X feet are corrected.** The raw SOMA→SMPL-X transfer produces plantarflexed (pointed-down) feet; a post-hoc ankle-alignment fix is applied during conversion.
> - **GPU0 on our box has uncorrectable ECC faults** → everything runs with `CUDA_VISIBLE_DEVICES=1`. Change this for your hardware.

---

## 0. Layout & conventions

```
WORKDIR = /simurgh/u/yaohe09/UniTokenws                     # repos, envs, scripts
DATA    = /simurgh2/datasets/LMM-Datasets/BONES-SEED        # all data + outputs
```
- **Repos** (clone under `WORKDIR`): `whole_body_tracking` (BeyondMimic fork, branch `seed-vae-sim2sim-pipeline`), `BFM-Zero`, `SOMA-X`.
- **Data lives on the fileserver** (`DATA`); repos/envs live on the working dir. Adjust both roots for your setup.

---

## 1. Prerequisites

| Need | Source / note |
|---|---|
| 1× CUDA GPU (we use L40S, sm_89) | headless EGL for rendering |
| **BONES-SEED dataset** | HuggingFace `bones-studio/seed` (gated — request access). Site: https://bones.studio/datasets/seed |
| **BFM-Zero checkpoint** | HuggingFace `LeCAR-Lab/BFM-Zero`, folder `new_model_for_training_code_inference/` (the one that ships the top-level `config.json` with an `env` block). Repo: https://github.com/LeCAR-Lab/BFM-Zero |
| **SMPL & SMPL-X neutral models** | gated (register at smpl.is.tue.mpg.de / smpl-x.is.tue.mpg.de). Need `SMPL_NEUTRAL.pkl`, `SMPLX_NEUTRAL.npz`. |
| **SOMA-X** | https://github.com/NVlabs/SOMA-X (SOMA↔SMPL converter) |
| **humenv** | `pip install git+https://github.com/facebookresearch/humenv.git` (BFM-Zero dependency, not on PyPI) |

---

## 2. Environments (three conda/venv)

```bash
# (A) somax  — SMPL/SMPL-X conversion + rendering  (python 3.12, torch 2.10)
bash WORKDIR/build_soma_env.sh
#   installs py-soma-x[smpl], chumpy, pyrender, imageio  (see script)
#   then symlink the gated body models into SOMA-X/assets:
ln -s <path>/SMPL_NEUTRAL.pkl   WORKDIR/SOMA-X/assets/SMPL/SMPL_NEUTRAL.pkl
ln -s <path>/SMPLX_NEUTRAL.npz  WORKDIR/SOMA-X/assets/SMPLX/SMPLX_NEUTRAL.npz

# (B) bfmzero — BFM-Zero MuJoCo tracking + state-action collection  (python 3.10, torch 2.5.1+cu124)
bash WORKDIR/build_bfmzero_env.sh
pip install "git+https://github.com/facebookresearch/humenv.git"   # requires approval; not on PyPI
#   BFM-Zero runs MuJoCo-only (no Isaac Sim): pip install -e BFM-Zero --no-deps

# (C) .venv6 — light env for Stage-1 CSV→artifacts + FK  (python 3.12, torch 2.10)
#   (only needs the whole_body_tracking stage utils + numpy/scipy/joblib)
```
Env sanity: `conda activate somax && python -c "import soma, smplx"`; `conda activate bfmzero && python -c "import humanoidverse, mujoco"`.

---

## 3. Fetch data & models

```bash
# BONES-SEED: robot CSVs (g1) + human BVH (soma_uniform) + shapes + metadata
#   -> DATA/g1/csv/<date>/*.csv           (142,220 clips)   [robot retarget]
#   -> DATA/soma_uniform/bvh/<date>/*.bvh (142,220 clips)   [human, SOMA rig]
#   -> DATA/soma_shapes/*.npz             [per-actor MHR shape params]
bash WORKDIR/fetch_soma_human.sh          # downloads+extracts soma_uniform/proportional
#   (g1.tar.gz fetched similarly; both are HF gated tarballs)

# BFM-Zero checkpoint -> DATA/bfmzero_model/new_model_for_training_code_inference/
bash WORKDIR/fetch_bfm_ckpt.sh
```
Clip naming: CSV stem == BVH stem == artifact name, e.g. `Relaxed_walk_forward_001__A057_M`.

---

## 4. The pipeline — two parallel tracks

Both tracks process the **same clip set**. Below is one batch of clips; §5 scales it over the whole corpus.

### Track A — Robot state–action (env: `.venv6` for prep, `bfmzero` for collection)

```
CSV ─(1)→ G1 artifact ─(2)→ ground-truth BFM-Zero motion pkl ─(3)→ (state,action) tuples
```

**(1) CSV → G1 artifact** (`whole_body_tracking/stage2/seed_to_artifacts.py`)
```bash
python stage2/seed_to_artifacts.py --src <csv_dir> --out <artifacts_dir> --limit 0
# -> <artifacts_dir>/<clip>:v0/motion.npz   (36-col G1 qpos @120fps -> artifact)
```

**(2) artifact → ground-truth BFM-Zero pkl** (`whole_body_tracking/stage3_sim2sim/to_bfmzero_motion.py`)
```bash
python stage3_sim2sim/to_bfmzero_motion.py --artifacts <artifacts_dir>/*:v0 --out gt.pkl --fps 30
# CPU forward-kinematics only. NO VAE. Produces the pkl BFM-Zero's motion-lib loads:
#   {clip: {root_trans_offset, pose_aa(T,30,3), dof(T,29), root_rot, fps}}
```
*Why ground-truth, not VAE-decoded:* a latent motion tokenizer we trained only to epoch-65 on 2k clips
smoothed away wrist (29.6°→6.6°) and waist motion. Tracking the raw retarget keeps full articulation.

**(3) BFM-Zero state–action collection** (`.../bfmzero_compare/collect_state_action.py`, env `bfmzero`)
```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python collect_state_action.py \
  --model-folder DATA/bfmzero_model/new_model_for_training_code_inference \
  --data-path gt.pkl --out-dir DATA/state_action --mode onpolicy       --simulator mujoco --max-steps 2000
#   repeat with --mode teacher_forced
python pack_state_action.py --root DATA/state_action --modes onpolicy teacher_forced
```
- **`onpolicy`** — closed-loop: log the states BFM-Zero actually visits while tracking + the action taken.
- **`teacher_forced`** — reset the robot onto the reference every control step; log `(reference state, action)`.
- Output: `DATA/state_action/{onpolicy,teacher_forced}/traj_<idx>_<clip>.npz`, plus `normalization.npz`,
  `manifest.json`, `dataset_stats.json`. Per-step keys (see §6).

### Track B — Human SMPL / SMPL-X (env: `somax`)

```
BVH ─→ SOMA pose ─→ SMPL & SMPL-X params ─→ foot-correction
```
One script does all of it: `SOMA-X/tools/batch_bvh_to_smpl.py` (uses `bvh_utils.py`, `bvh_to_soma_npz.py`,
`foot_align_correct.py`).
```bash
CUDA_VISIBLE_DEVICES=1 python -m tools.batch_bvh_to_smpl \
  --bvh-root DATA/soma_uniform/bvh --variant uniform \
  --out-root DATA/smpl_smplx --device cuda --shard 0 --num-shards 2   # run 2 shards
# -> DATA/smpl_smplx/uniform/<date>/<clip>.npz
```
Internals:
1. **BVH → SOMA pose** (`bvh_to_soma_npz.py`): the BONES-SEED BVH joints == `SOMALayer.public_joint_names`
   (exact order). Poses stored as **absolute** rotations (`absolute_pose=True, keep_root=True`).
   FK self-check vs the BVH: **0.78 cm** — the SOMA reconstruction is exact.
2. **SOMA → SMPL & SMPL-X** (SOMA-X `transfer_smpl_family_pose_parameters`). Identity-dependent setup
   (MHR eval, topology bridge, PoseInversion) is **hoisted once per identity** for speed.
3. **Foot correction** (`foot_align_correct.correct_pose`): the raw transfer plantarflexes the feet
   (pointed down). Fix = per-frame rotate each ankle (joints 7,8) so the foot bone (ankle→toe) points along
   the **SOMA source** foot direction (which is flat/correct). Validated: foot misalignment **35°→5°**.
   Every output npz is tagged `foot_corrected: True`.

> ⚠ Do **not** rely on SOMA-X's default fit knobs (`full_iters`, autograd, foot-weight) to fix the feet —
> tested, they don't. The post-hoc ankle alignment is required.

---

## 5. Scale-up orchestration (run the whole corpus)

`scale_up/process_batch.sh <offset> <count>` runs **one** contiguous batch of clips through **both tracks**
(matched increments), growing `DATA/state_action/` and `DATA/smpl_smplx/`. Resumable — every stage skips
existing outputs. `scale_up/loop.sh` chains batches until the corpus is exhausted:

```bash
bash WORKDIR/scale_up/loop.sh    # waits for any current jobs, then processes b000000, b002000, ...
```
- Batch = clips `[offset, offset+2000)` of the sorted CSV list.
- Per batch: Stage-1 (CPU) → ground-truth pkl (CPU) → **collect (bfmzero) + SMPL (somax) concurrently on GPU**.
- 2 SMPL shards is the sweet spot on one GPU (4 thrashed on our L40S). Progress: `scale_up/scale_up.log`,
  per-batch `scale_up/batch_*.log`, ledger `scale_up/completed_batches.txt`.

---

## 6. Output schema

### Robot — `state_action/{onpolicy,teacher_forced}/traj_*.npz` (per control step, len T)
| key | shape | meaning |
|---|---|---|
| `action` | (T,29) | BFM-Zero PD-target action **(the label)** |
| `z` | (T,256) | BFM-Zero task latent for the tracked frame |
| `qpos` / `qvel` | (T,36) / (T,35) | robot state (MuJoCo) |
| `obs_state` / `obs_privileged` | (T,64) / (T,P) | policy proprio / body-frame obs |
| `last_action` | (T,29) | previous action in the obs |
| `ref_dof_pos` / `ref_dof_vel` | (T,29) | reference joint targets |
| `ref_body_pos` / `ref_body_rots` | (T,B,3) / (T,B,4) | reference body poses |
| `alive`, `reward`, `ref_frame_idx` | (T,) | not-fallen flag, tracking reward, ref index |

Top level: `normalization.npz` (mean/std over qpos/qvel/obs_state/action/z/ref_*), `manifest.json`, `dataset_stats.json`.

### Human — `smpl_smplx/uniform/<clip>.npz`
| key | shape | meaning |
|---|---|---|
| `smpl_pose` / `smplx_pose` | (T,24,3) / (T,55,3) | axis-angle pose (foot-corrected) |
| `smpl_transl` / `smplx_transl` | (T,3) | root translation (m) |
| `smpl_verr_mean/max`, `smplx_verr_*` | scalar | per-vertex fit error (m) |
| `fps`, `T`, `clip`, `variant`, `foot_corrected` | — | metadata |

---

## 7. (Optional) Rendering — verification only

Triptychs `[ SMPL-X human | G1 reference | G1 executed ]` for spot-checking:
```bash
# ground-truth G1 rollouts for chosen clip indices (env bfmzero)
python .../batch_tracking_inference.py --model-folder <MF> --data-path gt.pkl \
  --out-dir <rolls> --simulator mujoco --indices 3 1516 ...
# render (uses SOMA-X/tools/render_smpl.py + .../render_bfmzero.py + SOMA-X/tools/stitch_triptych.py)
ROLLDIR=<rolls> bash WORKDIR/run_renders.sh <clip_list.txt>
# -> DATA/renders/paired/<clip>.mp4
```
Foot before/after demo: `SOMA-X/tools/render_foot_compare.py` → `renders/foot_correction_compare.mp4`.

**Self-contained robot renderer (no SOMA-X needed)** — `whole_body_tracking/stage2/render_g1_clip.py`
renders the `[ Reference | BFM-Zero executed ]` robot panels directly from a
`batch_tracking_inference` rollout npz using MuJoCo + EGL (the G1 scene XML), resampled to
real-time 30 fps. Example (960×480, matches `docs/*.mp4`):
```bash
# 1) rollout the clips (env bfmzero):  batch_tracking_inference.py --data-path <clips.pkl> --out-dir <rolls>
# 2) render each rollout:
MUJOCO_GL=egl python stage2/render_g1_clip.py --rollout <rolls>/rollout_0.npz \
  --out docs/sample_<clip>.mp4 --fps 30 --src-fps 50 --panels ref,exec
```
Sample outputs: `docs/sample_Neutral_kick_trash_001__A057.mp4`, `docs/sample_jog_squat_A492.mp4`.
(The human SMPL-X panel of the full triptych needs SOMA-X + the gated body models — Track B.)

---

## 8. Results on the validation subset (2004 robot / 1420 human clips)

| Dataset | Size |
|---|---|
| Robot state–action | **1,417,918 transitions**, mean survival **0.992** (on-policy + teacher-forced) |
| Human SMPL/SMPL-X | **1420 clips**, foot-corrected, fit error ~SMPL 4 cm / SMPL-X 3.5 cm |

The `scale_up/loop.sh` extends both to the full 142,220-clip corpus with the identical pipeline.

---

## 9. Gotchas / lessons

- **VAE-decoded reference is bad data** — always track the ground-truth retarget for the robot side.
- **SMPL-X feet need the ankle-alignment fix** — the default SOMA→SMPL-X transfer plantarflexes them.
- **BFM-Zero checkpoint**: use `new_model_for_training_code_inference/` (has the env config), not `model/`.
- **MuJoCo offscreen render caps at 480 px** — use `--render-size 480`.
- **One healthy GPU**: 2 SMPL shards, not 4 (memory-bandwidth thrash). `CUDA_VISIBLE_DEVICES=1` here (GPU0 ECC-dead).
- **`pgrep` in wait-loops**: match `python.*<name>`, not the bare name, or it matches the script's own text.
```
