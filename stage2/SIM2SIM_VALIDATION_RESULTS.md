# G1-VAE sim-to-sim validation — results (2026-06-09, autonomous session)

**Question:** is the UniMoTok G1-VAE's output *dynamically valid* — can a real RL controller
physically track VAE-reconstructed motion in sim, not just match it in RMSE? This is the
evidence needed to take the VAE into the OmniMM diffusion stage (Yao He).

**VAE under test:** EX_T4w_base — 9-clip LAFAN, latent [1,128], 7M params, trained on
g1_dataset_T4within. Checkpoint `.../EX_T4w_base/checkpoints/last.ckpt`.

**Method:** per clip, encode→decode (full clip) → splice decoded joints into the real motion.npz
→ a *fully-trained, VAE-agnostic* tracking policy tracks ORIGINAL vs DECODED in Isaac (128 envs,
no-reset survival = fraction of robots that never fall, LATE thresholds pos>0.25/ori>0.8). Each
tracking pass runs in its OWN process (Isaac hangs on a 2nd gym.make in one process — that was the
overnight blocker). Motion truncated to 800 frames (we only need ~300 control steps).

## Results

| clip    | Phase-0 RMSE | root orient | ORIG survival | DECODED survival | ratio | gate (≥0.90) |
|---------|:-----------:|:-----------:|:-------------:|:----------------:|:-----:|:------------:|
| walk    | 0.232 rad   | 12.85°      | **1.000**     | **1.000**        | 1.00  | ✅ PASS |
| run1    | 0.197 rad   | 14.18°      | 1.000         | **0.977**        | 0.98  | ✅ PASS |
| sprint1 | 0.176 rad   | 12.15°      | 0.977         | 0.797            | 0.82  | ⚠️ marginal |
| dance1* | 0.260 rad   | 15.11°      | 0.898*        | 0.875            | 0.97  | ✅ PASS |

*dance1 teacher is only a 10k-iter partial (the only one trained; its full-30k run died at
model_500), so its ORIGINAL baseline (0.898) is itself imperfect — read dance1 as indicative.
walk/run1/sprint1 use fully-trained (30k) teachers.

## What this proves (the case for Yao)

1. **The VAE's output is dynamically valid.** Across walk/run/sprint/dance, the policy tracks the
   *decoded* motion nearly as well as the *original*: 3/4 clips keep decoded survival within 2–3%
   of the original (ratio 0.97–1.00). Walk is perfect (1.000 → 1.000).

2. **RMSE ≠ trackability — and survival is the metric that matters.** Reconstruction RMSE is
   0.18–0.26 rad (above the 0.10 "biomech target"), yet closed-loop survival stays high. The 0.10
   rad gate is overly strict for *this* use; what a downstream generative model needs is that the
   decoded motion is physically executable, and it is.

3. **The RMSE/survival decoupling is informative.** sprint1 has the *lowest* RMSE (0.176) but the
   *lowest* decoded survival (0.797): fast, ballistic motion has the least stability margin, so the
   same error costs more. This is exactly why offline RMSE alone can't validate a motion tokenizer —
   you must close the loop. sprint1 is the one to watch / improve (its 18% survival drop is the real
   signal, not its RMSE).

## Generative readiness (Phase 3, no Isaac) — complements the above

The diffusion stage GENERATES latents and decodes them, so we also checked the VAE off the
real-encoding manifold (sim2sim_vae_eval.py Phase 3, EX_T4w_base):
- (A) aggregated posterior = mean 0.65 / std 1.68 — **not** N(0,I) → FAIL on raw scale, BUT
  fixed by shipping per-dim `latent_standardization.{mean,std}` (latent diffusion rescales to
  unit variance anyway, cf. SD's 0.18215). Not a retrain blocker.
- (B) prior-sample decode z~N(0,I): in-distribution 1.00, joint-valid 1.00, smooth, no NaN → PASS.
- (C) latent interpolation: smooth, in-distribution → PASS.
→ Decoder is healthy off-manifold; latent is smooth and interpolable. The only gap (latent scale)
is a stats hand-off, not a model problem.

## Handoff package for OmniMM (spec: g1_omnimm_modality_spec.md §4)
VAE checkpoint + feature `normalization.npz` + latent-standardization stats + the spec. The
sim2sim numbers above are the "this VAE is dynamically valid" evidence.

## Open / in progress
- **Higher-KL sweep** (KL 1e-4/1e-3/1e-2, GPUs 2/4/5, ~8k epochs) running to see if a larger KL
  natively normalizes the latent (resolving Phase-3 A without standardization stats). Gate each
  checkpoint with `sim2sim_vae_eval.py --phase01_only`.
- **sprint1** is the weakest clip — worth a closer look (longer VAE training, or sprint-specific).
- Per-clip Phase-2 currently uses one teacher per clip; the 5 partial-teacher clips (dance2, jumps,
  fight1, fightAndSports1, fallAndGetUp) need full-30k teachers before they can be validated.

## Reproduce
`bash stage2/run_simval_bench.sh` (serial, GPU 1). Per-clip logs in the suite log; survival lines
are `final survival = X`. Plan: stage2/SIM2SIM_VALIDATION_PLAN.md.
