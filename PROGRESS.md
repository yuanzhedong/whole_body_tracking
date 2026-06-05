# Progress Log — whole_body_tracking

Reproducing BeyondMimic end-to-end (Fig 7) and integrating G1 robot motion into OmniMM
(Unified Large Motion Model). Reverse-chronological; each entry is a concrete deliverable.

---

## 2026-06-05 — G1 → OmniMM pipeline: Steps 1 & 2

**Step 1 — G1 motion exporter** (`stage2/export_g1_motion.py`, commit `b5708ba` + update)
- Exports `artifacts/*/motion.npz` (BeyondMimic reference motions) to a UniMoTok-compatible
  G1 dataset: 41-D velocity representation (6-D root rot + root lin/ang vel + 29 joint angles),
  heading-canonical, with FK ground truth (`body_pos_w`, `joint_pos`) preserved for eval.
- Clip-based train/val/test split (no window leakage), category metadata, mirror-aug table.
- 12 clips (AMASS TODO'd) → 36,795 train + 3,892 val + 273 test windows @ 20 fps / 128-frame windows.
- Dataset at `stage2/out/g1_dataset/` with `manifest.json` + `normalization.npz`.

**Step 2 — z-up → y-up conversion** (`--to_yup` flag, same file)
- Implements basis change `T = [[1,0,0],[0,0,1],[0,-1,0]]` (Isaac z-up → OmniMM y-up).
- Heading computation in y-up frame: yaw about +y, `roty` strip, FK ground truth also converted.
- `--verify_only --to_yup` prints mean root height + range checks before any files are written.
- AMASS clips excluded with TODO — need larger corpus retargeted via `retargeting/`.

**Per-clip quality eval** (`stage2/eval_tracking_quality.py`)
- Runs Stage-0 policy on each clip, records survival rate + E_mpbpe_mm per clip.
- Output JSON feeds `export_g1_motion.py --quality_json` to filter low-quality clips.
- Run: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python stage2/eval_tracking_quality.py --teacher_ckpt <ckpt>`

**G1 → OmniMM spec** (`stage2/g1_omnimm_modality_spec.md`, commit `a667d38`)
- Full build-order table, representation rationale (6-D rot, velocity rep, 41-D), and OmniMM
  branch hand-off. Grounded in real repo (29 DoF, pelvis/torso_link, 50 Hz, npz schema).

---

## 2026-06-05 — UniMoTok standalone + Fig-7 VAE analysis

**Submodule** (`UniMoTok/`, commit `db0246e`)
- Registered `Juzezhang/UniMoTok` as a git submodule, pinned to branch `wbt-integration`
  (off `feat/biomechanics_tokenization` — the *main* branch is incomplete, `models/` is
  git-ignored there).
- Key gotcha: always use `feat/biomechanics_tokenization` or `wbt-integration`; never `main`.

**MLD VAE standalone smoke** (`UniMoTok/smoke_mld_vae.py`)
- Instantiates `MldVaeBiomechanics` (22.4M params, latent [1,256], nfeats 49) from the exact
  YAML params; overfits a synthetic `[8,128,49]` batch (no cluster data needed).
- Result: recon loss 0.240 → 0.045 (5.3× lower). **SMOKE_PASS.**
- Cluster datasets (`/simurgh2/…`) are absent here — reproducing their published numbers
  requires data from the Stanford cluster.

**Why UniMoTok ≠ drop-in Fig-7 VAE** (`stage2/unimotok_vs_figure7.md`, commit `db0246e`)
- Three axes of mismatch: motion-reconstruct vs action-from-proprio; offline-windowed vs
  online-per-step; reconstruction vs DAgger-in-sim.
- Two integration paths: (A) generate-then-track (natural); (B) arch-transplant (faithful, heavy).
- *This analysis is for the BeyondMimic Fig-7 control VAE — not relevant to the OmniMM plan
  where UniMoTok is used as designed.*

---

## 2026-06-04 — Stage-2 Phase-1 VAE distillation + verification (BeyondMimic Fig 7)

**VAE model** (`stage2/vae_model.py`, `stage2/distill_vae.py`, commit `129ba83`)
- Faithful BeyondMimic Table S6: MLP [2048,1024,512] ELU, latent 32, β=0.01, lr 5e-4, accum 15.
- Encoder input: command + anchor pose error (67-D ref terms). Decoder: ref-latent + proprio → action.
- DAgger distillation in `Tracking-Flat-G1-v0` against the walk-30k teacher.

**Verification gates G1–G4** (`stage2/verify_vae.py`, commit `b12580c`)
- 10k-iter distillation result (as of 2026-06-05): recon MSE 0.92 ✅, latent active 32/32 ✅,
  z-ablation 0.14 ✅, **closed-loop survival 0.456 vs teacher 0.999 ❌**.
- Verdict: machinery proven end-to-end; undertrained (recon still 1.34 at 10k). Needs ~50-100k iters.
- Gate G2 (closed-loop) is the hard blocker for moving to the diffusion stage.

---

## 2026-05-30 — AMASS retargeting pipeline

**Retargeting** (`retargeting/`, commit `e0e907f`)
- AMASS SMPL-H → G1 via pre-retargeted HumanML3D data from HuggingFace (`yuanzhedong/wbt_fix`).
- Root height fix: +0.793 m offset to avoid floor-clipping (from the G1 pelvis height at rest).
- Pipeline reuses BeyondMimic's stock `csv_to_npz.py` after converting to the expected CSV format.
- GMR-based retargeting deferred (needs more setup, offline).

---

## 2026-05-30 — Tracking eval tooling + Stage-0 verification

**Deterministic eval** (`tools/eval_tracking.py`, commit `ccdfe85`)
- Per-clip survival rate + E_mpbpe / E_mpjpe / E_anchor via the Isaac Sim env, headless.
- Run results logged to W&B (`cs224n-robustqa` org, `Motions` registry artifact type).

**Stage-0 verified results** (W&B reports, 2026-05-30)
- LAFAN1 test suite: **98.9% survival** avg.
- AMASS test suite: **94.9% survival** avg.
- 8-motion BeyondMimic suite: **~90% survival** avg (dance/fight clips lower, locomotion higher).
- Teacher: `walk1_subject1` model_29999.pt, 50 Hz, 2048 envs, 30k steps.

---

## 2026-05-30 — Local tooling + environment setup

**Environment** (no commit; local config)
- Isaac Sim 4.5 + Isaac Lab 2.1; Python 3.10 uv venv (`.venv/`).
- `OMNI_KIT_ACCEPT_EULA=YES CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` required.
- GPUs 0,3 = Blackwell (no torch kernels) — use 1,2,4,5 (4090) only.
- RTX renderer segfaults on driver 595 (needs ~535); use `--headless` for all training/eval.

**Run/render scripts** (`run.sh`, `render_policy.sh`, commit `9e9aac9`)
- `run.sh`: single-command training launch with correct env vars.
- `render_policy.sh`: headless rollout → frames → video (RTX path guarded; stagger launches to
  avoid URDF→USD race on simultaneous multi-clip renders).

---

## Outstanding TODOs

| item | status | notes |
|---|---|---|
| AMASS clips in G1 dataset | TODO | need larger retargeted corpus; `amass_*` skipped in exporter |
| z-up→y-up visual verify | TODO | run `--verify_only --to_yup`, then render one clip to confirm |
| Per-clip quality filter | TODO | run `eval_tracking_quality.py` on 4090, then re-export |
| Stage-2 VAE longer training | TODO | need ~50-100k iters for G2 gate to pass |
| G1-VAE config + datamodule | TODO | `config_g1_mldvae.yaml` + `G1DataModule` (Step 2 of OmniMM pipeline) |
| OmniMM branch wiring | BLOCKED | needs Yao He's OmniMM repo |
| Stage-2 diffusion (Phase 2–4) | BLOCKED | needs G2+G3 gates to pass first |
