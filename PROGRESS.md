# Progress Log — whole_body_tracking

Reproducing BeyondMimic end-to-end (Fig 7) and integrating G1 robot motion into OmniMM
(Unified Large Motion Model). Reverse-chronological; each entry is a concrete deliverable.

---

## 2026-06-06 20:18 — Wakeup check: run policy ~3h left, v2 val locked at 3.334

- Run policy: RUNNING, ETA ~03:05 (~23:23 finish)
- G1-VAE v2: epoch=9138, train=0.844, val=**3.334** (locked at 3.334 for 4000+ epochs — stable plateau, not diverging; gap=+0.28 from best epoch=1079)
- Post-watcher: sleeping

---

## 2026-06-06 19:47 — Wakeup check: run policy ~3.7h left, v2 val plateau

- Run policy: RUNNING, ETA ~03:44 (~23:30 finish)
- G1-VAE v2: epoch=7280, train=0.914, val=**3.334** (plateaued stable at ~3.33 since epoch 5000; gap=+0.28 from best, below kill threshold 3.5)
- Post-watcher: sleeping

---

## 2026-06-06 19:16 — Wakeup check: all jobs running, ~4.4h to run policy exit

- Run policy: RUNNING, ETA ~04:26 (~23:40 finish)
- G1-VAE v2: epoch=5389, train=1.034, val=3.336 (gap from best=+0.28, still below kill threshold 3.5)
- Post-watcher: sleeping, queued jobs still pending
- No output files yet

---

## 2026-06-06 18:14 — G1-VAE v2 best checkpoint confirmed, data bottleneck proven

**G1-VAE v2 best checkpoint: `epoch=1079.ckpt` (val=3.054)**

Full comparison (Phase 0 RMSE on OOD val/test clips):

| model | params | val loss | jt RMSE (fall) | jt RMSE (dance) |
|---|---|---|---|---|
| v1 epoch 564 | 22.4M | 3.481 | 0.418 rad | 0.359 rad |
| **v2 epoch 1079** | 7M | **3.054** | 0.424 rad | **0.351 rad** |
| v2 epoch 3569 | 7M | 3.227 | 0.435 rad | 0.361 rad |
| target | — | — | **< 0.10 rad** | **< 0.10 rad** |

**Conclusion: data bottleneck proven.** Val loss improved 12% (3.054 vs 3.481) but RMSE is
essentially unchanged — both models fail equally on OOD categories (fall/dance) because they've
never seen those motion types. Neither model size nor regularization helps. Need diverse clips.

**v2 current state** (epoch 3569): past best, slowly overfitting (gap 3.054→3.227). Will continue
to 20k since v3 will replace it after new category data arrives. Best ckpt = epoch=1079.

**Still waiting:** run policy ETA ~5h. Post-watcher queued (quality eval → re-export → verify → sim2sim Phase 2 → sprint).

---

## 2026-06-06 17:42 — G1-VAE v2 vs v1 comparison + wakeup check

**G1-VAE v2 (epoch 1809, ~30 min wakeup check):**
- WandB run `en071hgo`, train=1.856, val=**3.138** (stable, not diverging)
- v2 val=3.138 vs v1 best val=3.481 → **10% lower** — smaller model reduces overfitting

**Phase 0 RMSE (v2 epoch=1809 vs v1 epoch=564):**

| clip | category | v1 RMSE | v2 RMSE | conclusion |
|---|---|---|---|---|
| lafan_fallAndGetUp | fall (OOD) | 0.418 rad | 0.435 rad | ~same |
| dance1 | dance (OOD) | 0.359 rad | 0.361 rad | ~same |

**Key insight: data bottleneck confirmed.** Both models hit the same reconstruction floor because
neither has seen fall/dance clips in training (only walk-heavy 12 clips). Smaller model reduces
overfitting but can't generalize to unseen categories. Need run/sprint/jump/dance/fight clips.

**All Isaac jobs remain blocked** until run policy (PID 3484795) finishes in ~5.5h.
Post-watcher will then fire: quality eval → re-export → Stage-2 verify → sim2sim Phase 2 → sprint.

---

## 2026-06-06 — Autonomous session: sim2sim pipeline + G1-VAE v2

**sim-to-sim validation pipeline** (`stage2/sim2sim_vae_eval.py`, commit `21ea483`)
- **Phase 0+1 fully proven** (no Isaac): encode→decode→splice-into-motion-npz works cleanly.
  See `stage2/VAE_VALIDATION.md` for detailed results.
- **Phase 2 (Isaac tracking comparison) blocked**: Isaac's `DriverShaderCacheManager` is a
  machine-wide singleton held by the run policy. New Isaac processes can't initialize while it
  runs. Will execute Phase 2 when run policy finishes (~6h from now).
- Key fix chain: CUDA device→CPU for VAE, `import whole_body_tracking.tasks` at module level,
  Phase 1 switched from FK-replay to joint-splice (avoids Isaac entirely for decode→npz).

**G1-VAE v2 training — RUNNING** (GPU 5, UniMoTok, `experiments/biomechanics_tokenizer/G1_MldVAE_v2_smallreg/`)
- Smaller model to fight overfitting: num_layers=5, ff_size=1024, latent=[1,128] (~7M params vs 22.4M).
- Stronger regularization: KL=5e-5, dropout=0.15, weight_decay=1e-4.
- step_size=32 (was 64) → 1162 train windows (2× more than v1).
- Same 12-clip dataset (y-up, no quality filter yet). Quality filter pending run policy finishing.
- WandB: `cs224n-robustqa/g1-motion-tokenizer` (new run auto-created).

**G1-VAE v1 — STOPPED** (was overfitting from epoch ~600 onwards)
- Best checkpoint: `epoch=564.ckpt`, val_loss=3.481 (train=0.87→val=3.48, gap=2.6 = severe overfitting).
- Root cause: 12 clips / 581 train windows insufficient for 22.4M-param Transformer.

**Run policy (lafan_run1_subject2)** — TRAINING, GPU 1, ~6h remaining
- `logs/rsl_rl/g1_flat/lafan_run1_subject2_full30k/`, WandB: W&B auto-logged.
- Sequential watcher will launch sprint, jump, dance, fight after it finishes.

**Zombie process cleanup — 2026-06-06**
- Killed 4 csv_to_npz Isaac processes from 5–6 days ago (holding ~6 GB on GPU 5).
- Root cause of all subsequent Isaac failures: `DriverShaderCacheManager` held by zombies.
- After cleanup, run policy holds the singleton exclusively.
- `fuser /dev/nvidiaN` is the correct command to identify processes holding a GPU.

---

## 2026-06-05 — G1-VAE training + new tracking policies launched

**G1-VAE v1 training — RUNNING** (GPU 4, UniMoTok, `experiments/biomechanics_tokenizer/G1_MldVAE_v1/`)
- Stack: `MldVaeBiomechanics` (22.4M params, latent [1,256], 9-layer encoder_decoder, nfeats=41).
- Data: 12 LAFAN1/walk clips, 36,795 train windows @ 20 fps / 128-frame, y-up, **no quality filter yet**.
- 50k epochs (9 batches/epoch ≈ 450k gradient steps), AdamW lr=1e-4, cosine decay.
- WandB: https://wandb.ai/cs224n-robustqa/g1-motion-tokenizer/runs/z6ykfmkx
- Early result at epoch 48: val loss 5.657 and **decreasing** (from 5.78 at epoch 40). Training confirmed working.
- Checkpoints at `UniMoTok/experiments/biomechanics_tokenizer/G1_MldVAE_v1/checkpoints/`.

**Bugs fixed during launch** (all on `wbt-integration` branch, commits `e6fb9bf`, `83e8aae`):
- `metrics/base.py`: lazy-import so `METRIC.TYPE=[]` doesn't pull fasttext/smplx/NLP at startup.
- `biomechanics_tokenizer.py`: generalize recons_slice, root_orient_loss, root_orient_velocity_loss
  to arbitrary feature dims (was hardcoded to 49/52; G1 is 41).
- `callback.py`: replace RichProgressBar with TQDMProgressBar (Rich crashes headless on sanity check).
- `G1DataModule.py`: new datamodule reading stage2/out/g1_dataset_yup/{train,val,test}/*.npz.
- Deps installed: pandas, loguru, matplotlib, opencv-headless, smplx, rich, wandb, CUDA torch 2.5.1+cu124.

**Stage-2 VAE 50k iter distillation — RUNNING** (GPU 1, walk teacher)
- At 7300/50000 iters, recon 1.99 and declining (21.7 it/s, ~31 min to completion).
- Output: `stage2/out/vae_walk_50k.pt`.

**Per-clip tracking quality eval — RUNNING** (GPU 5)
- Evaluating 12 clips with walk teacher; output → `stage2/out/track_quality.json`.
- When done: auto-trigger re-export with `--quality_json` + `--to_yup` → `g1_dataset_yup_filtered/`.
- Then: auto-trigger G1-VAE v2 training on filtered data.

**Sequential new tracking policies — QUEUED** (launches on GPU 1 after Stage-2 finishes)
- Queue: lafan_run1_subject2, lafan_sprint1_subject2, lafan_jumps1_subject1, lafan_dance1_subject1.
- Each: 30k iter, 2048 envs, W&B registry. Will take ~2h each.

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
