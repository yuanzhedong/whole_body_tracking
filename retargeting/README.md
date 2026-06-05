# `retargeting/` — AMASS → Unitree G1 → tracking policy

Extends the motion source for the BeyondMimic tracking pipeline from the bundled LAFAN1 clips to
the much larger **AMASS** human-motion corpus, then **reuses the existing pipeline unchanged**
(`scripts/csv_to_npz.py` → W&B registry → `scripts/rsl_rl/train.py` → `tools/eval_tracking.py` /
`render_policy.sh`).

## Two ways to get a G1 motion from AMASS

```
                         ┌─ (A) pre-retargeted (no credentials)  ── implemented here
AMASS (human SMPL) ──────┤
                         └─ (B) our own retargeting via GMR       ── deferred (needs gated data)
```

### (A) Pre-retargeted AMASS (default, no license-gated downloads)
Community dataset **`fleaven/Retargeted_AMASS_for_robotics`** (CC-BY-4.0) already provides AMASS
retargeted to the G1 in *this repo's exact format*. `hf_to_csv.py` downloads one clip and writes a
repo-format CSV.

```bash
# browse walk clips
python retargeting/hf_to_csv.py --list-walks
# convert one clip -> CSV  (prints the fps to use next)
python retargeting/hf_to_csv.py \
    --file "g1/ACCAD/s007/QkWalk1_poses_120_jpos.npy" \
    --out  retargeting/out/amass_accad_qkwalk1.csv
```

**Format note (important):** the HF `.npy` is `[N,36]` = `root_xyz(3) + root_quat_xyzw(4) + 29 joints`,
with the **29 joints in the same order** as `scripts/csv_to_npz.py`. The stored root **z is offset by
−0.793 m** (per the dataset's own `g1/visualize.py`), so the converter **adds 0.793** to lift the
pelvis to its true standing height (~0.8 m, matching LAFAN1). fps is parsed from the filename
(`_poses_120_jpos` → 120). Without the height fix the robot would be buried in the floor.

### (B) Our own SMPL→G1 retargeting via GMR (deferred — see `gmr_setup.md` once added)
`github.com/YanjieZe/GMR` retargets raw AMASS (SMPL-X) → G1 in its own CPU venv. It needs
**license-gated downloads you must fetch yourself** (free academic registration):
AMASS `.npz` (amass.is.tue.mpg.de) + SMPL-X body models (smpl-x.is.tue.mpg.de). GMR emits a
30-joint trajectory; a small remap to this repo's 29-joint order produces the same CSV, after which
everything downstream is identical. Build this only if pre-retargeted quality (below) is insufficient
or you need motions not already in the HF set (e.g. from your own video).

## Feed it into the existing pipeline
```bash
# NOTE: pin a 4090 (the Isaac .venv has no Blackwell kernels); idx 1 may be busy with a training run.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES \
  .venv/bin/python scripts/csv_to_npz.py \
  --input_file retargeting/out/amass_accad_qkwalk1.csv --input_fps 120 \
  --output_name amass_accad_qkwalk1 --output_fps 50 --headless &
# csv_to_npz never exits in --headless: it uploads to the W&B motions registry, then loops.
# Confirm the artifact via the W&B API, copy /tmp/motion.npz, then kill the process.

# then train exactly like any LAFAN1 motion:
./run.sh scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name cs224n-robustqa/wandb-registry-motions/amass_accad_qkwalk1 \
  --headless --logger wandb --log_project_name beyondmimic-tracking --run_name amass_walk
```

## Quality assessment (assess-first)
Pre-retargeted quality varies by source/method, so **measure before trusting**:
```bash
python retargeting/assess_motion.py --npz /tmp/motion.npz --baseline <lafan_walk.npz>   # metrics + verdict
python retargeting/viz_motion.py    --npz /tmp/motion.npz --out retargeting/out/amass_walk.mp4
```
`assess_motion.py` reports ground penetration, foot skate, jitter, joint-limit violations, and root
sanity, with a pass/fail verdict ("quality good → GMR optional" vs "artifacts → consider GMR").
`viz_motion.py` renders the motion with the repo's decoupled Sim-6.0 renderer (interactive Isaac
rendering segfaults on this box's driver).

## Files
- `hf_to_csv.py` — download a pre-retargeted G1 clip and write a repo-format CSV (+ validation).
- `assess_motion.py` — quality metrics + verdict on the `csv_to_npz` output npz.
- `viz_motion.py` — render a motion npz to MP4 via `tools/render_rollout_sim6.py`.
- `tests/test_convert.py` — format/contract tests (`pytest retargeting/tests/`).
- `out/`, `data/` — generated CSVs and downloaded clips (git-ignored).

## Findings (first end-to-end run: ACCAD `QkWalk1`, a brisk walk)

Validated the full chain `hf_to_csv → csv_to_npz → registry → train → eval → render` on one clip:

- **Format fix that mattered:** the HF root z is stored offset by −0.793 m; without adding it back
  the robot trains buried in the floor. `hf_to_csv.py` applies it; tests lock it in.
- **Quality vs the proven LAFAN1 walk** (`assess_motion.py`, baseline-relative): ground
  penetration, foot-skate, and joint-limit overshoot are **on par** with LAFAN1 (which trains to
  ~99% success); jitter is ~3× (it's a fast clip). Render shows a **plausible upright walk, feet on
  the floor**. → pre-retargeted quality is **usable**; GMR optional for this clip.
- **Pipeline reuse works unchanged:** a 300-iter smoke trained, checkpointed, and evaluated via the
  stock scripts (success 47.6%, E_mpbpe 68.6 mm at 300 iters — early but on the LAFAN1 trajectory;
  full quality needs ~30k iters like any motion).

**Verdict:** for clips whose `assess_motion.py` metrics track the LAFAN1 baseline, use the
pre-retargeted HF data directly. Build GMR (section B) only for motions absent from the HF set,
your own video-sourced motion, or if a clip's assessment flags real artifacts.

## Licensing
AMASS and the retargeted derivatives are **non-commercial academic**; propagate the original
AMASS sub-dataset licenses and cite AMASS + SMPL-X. Gated data / body models are git-ignored and
must never be committed or redistributed.
