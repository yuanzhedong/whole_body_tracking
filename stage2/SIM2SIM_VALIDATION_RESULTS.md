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

## Results — all 9 LAFAN clips

| clip            | Phase-0 RMSE | ORIG surv | DECODED surv | ratio | gate (≥0.90) | teacher |
|-----------------|:-----------:|:---------:|:------------:|:-----:|:------------:|:-------:|
| walk            | 0.232 rad   | 1.000     | **1.000**    | 1.00  | ✅ PASS | 30k |
| run1            | 0.197 rad   | 1.000     | 0.977        | 0.98  | ✅ PASS | 30k |
| dance2          | 0.275 rad   | 0.992     | 0.984        | 0.99  | ✅ PASS | 10k |
| dance1          | 0.260 rad   | 0.898     | 0.875        | 0.97  | ✅ PASS | 10k |
| jumps1          | 0.201 rad   | 0.961     | 0.875        | 0.91  | ✅ PASS | 6k  |
| sprint1         | 0.176 rad   | 0.977     | 0.797        | 0.82  | ⚠️ marginal | 30k |
| fight1          | 0.225 rad   | 0.961     | 0.742        | 0.77  | ❌ FAIL | 10k |
| fightAndSports1 | 0.226 rad   | 1.000     | 0.414        | 0.41  | ❌ FAIL | 10k |
| fallAndGetUp1   | 0.255 rad   | 0.969     | 0.219        | 0.23  | ❌ FAIL | 10k |

(128 envs, no-reset survival, LATE thresholds. teacher = tracking-policy iters; 30k = fully trained,
6k–10k = partial. ORIG survival is the teacher's own ceiling — all teachers track their original
clip at ≥0.90, so failures below are the VAE reconstruction, not a broken teacher.)

## What this proves (the honest case for Yao)

1. **The VAE is dynamically valid for locomotion + dance — 5/9 clips PASS.** walk (1.00), run1
   (0.98), dance2 (0.99), dance1 (0.97), jumps1 (0.91): the policy tracks the *decoded* motion within
   a few % of the *original*. Walk is perfect (1.000→1.000). For the motions a versatile-control
   model spends most of its time on, the VAE's output is physically executable.

2. **It breaks on the most dynamic / contact-rich motions — fight & fall-and-get-up FAIL.** fight1
   (0.77), fightAndSports1 (0.41), fallAndGetUp (0.23): decoded survival collapses even though the
   teacher tracks the *original* fine (0.96–1.00). These motions have ballistic, contact-critical
   phases (strikes, ground push-offs, the get-up) where a small joint error compounds into a fall.

3. **RMSE ≠ trackability — survival is the metric that matters (the headline).** The clincher:
   **fallAndGetUp (RMSE 0.255) and dance1 (RMSE 0.260) have essentially identical reconstruction
   error, but decoded survival of 0.22 vs 0.88 — a 4× gap RMSE is blind to.** Likewise sprint1 has
   the *lowest* RMSE (0.176) yet a marginal 0.82 ratio. Offline RMSE cannot validate a motion
   tokenizer; you must close the loop. This is the core argument for using sim2sim survival (not the
   0.10-rad biomech RMSE target) as the VAE-quality gate.

4. **Caveat (honest):** the 4 failing/marginal-adjacent clips (fight1, fightAndSports1, fallAndGetUp,
   and dance1/dance2/jumps1) use 6k–10k partial teachers. A partial teacher has a narrower competence
   basin and may be more fragile to a slightly-off decoded reference, so part of the fight/fallGetUp
   drop could be teacher fragility rather than pure VAE error. dance2 PASSES at 0.99 with a 10k
   teacher, so partial≠doomed — but to fully attribute the failures to the VAE we need full-30k
   teachers for fight1/fightAndSports1/fallAndGetUp. That is the cleanest next experiment.

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

## Higher-KL sweep — RESULT: KL is NOT the lever for latent normalization (negative result)
Trained KL ∈ {1e-4, 1e-3, 1e-2} on T4within (vs base 5e-5), gated each with the Phase-3 gate.
At matched epoch (~645):

| KL    | aggStd (want ~1) | aggMu (want ~0) | val RMSE |
|-------|:----------------:|:---------------:|:--------:|
| 5e-5 (base) | 1.68       | 0.65            | ~0.40    |
| 1e-4  | 1.50             | 0.25            | 0.358    |
| 1e-3  | 1.50             | 0.25            | 0.357    |
| 1e-2  | 1.48             | 0.26            | 0.356    |

Raising KL from 5e-5→1e-4 helps a little (aggStd 1.68→1.50, aggMu 0.65→0.25), then **saturates**:
1e-4, 1e-3, 1e-2 are indistinguishable (aggStd ~1.49) despite a 100× KL range, and reconstruction
is unchanged (~0.357). The latent scale plateaus at ~1.5, never reaching N(0,I). **Conclusion: KL
weight cannot normalize this latent** — the Phase-3 (A) gap must be handled by the per-dim
`latent_standardization` stats (which latent diffusion applies anyway, cf. SD 0.18215), NOT by
retraining. Sweep kept running to confirm the plateau holds at convergence / surface any late
recon-vs-KL tradeoff; logs /tmp/kl_genready.log.

## Open / next
- **Full-30k teachers for fight1 / fightAndSports1 / fallAndGetUp** — the single cleanest next
  experiment: disentangles "partial teacher is fragile" from "VAE genuinely can't reconstruct this
  motion well enough." If decoded survival jumps with a strong teacher → teacher was the issue; if it
  stays low → the VAE needs work on dynamic/contact motion.
- **Improve the VAE on dynamic/contact motion** (if (above) confirms it's the VAE): per-joint
  normalization, velocity/contact-aware loss, or up-weighting the hard clips. RMSE won't show the
  gain — re-run this sim2sim suite to measure it.
- **sprint1** marginal (0.82) — same dynamic-motion theme.
- **sprint1** is the weakest clip — worth a closer look (longer VAE training, or sprint-specific).
- Per-clip Phase-2 currently uses one teacher per clip; the 5 partial-teacher clips (dance2, jumps,
  fight1, fightAndSports1, fallAndGetUp) need full-30k teachers before they can be validated.

## Reproduce
`bash stage2/run_simval_bench.sh` (walk/run1/sprint1/dance1) and `bash stage2/run_simval_bench2.sh`
(dance2/jumps1/fight1/fightAndSports1/fallAndGetUp), serial on GPU 1. Survival lines are
`final survival = X` in the suite logs. Plan: stage2/SIM2SIM_VALIDATION_PLAN.md.
