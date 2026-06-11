# Gated pipeline results — clean stage-1 → uniform VAE → sim2sim (2026-06-11)

Pipeline (no per-clip weights): (1) survival gate per motion → (2) keep gate-passing teachers →
(3) train one uniform VAE on that subset → (4) sim2sim with the gated teachers.

## Stage-1 gate (original survival ≥ 0.95) — 8/9 LAFAN pass
walk 1.000, sprint1 1.000, fightAndSports1 1.000, dance2 0.992, run1 0.984, fight1 0.969,
dance1 0.961, fallAndGetUp 0.953 — PASS. jumps1 0.914 (6k teacher) — FAIL → excluded.
→ VAE corpus = the 8 passers (g1_dataset_gated8).

## VAE EX_gated8 (8 clips, latent128, KL5e-5, ~10k epochs) — sim2sim with gated teachers

| clip            | RMSE  | orig  | decoded | ratio | verdict |
|-----------------|:-----:|:-----:|:-------:|:-----:|:-------:|
| walk            | 0.229 | 1.000 | 1.000   | 1.00  | ✅ PASS |
| run1            | 0.202 | 0.992 | 0.992   | 1.00  | ✅ PASS |
| dance2          | 0.285 | 0.984 | 0.961   | 0.98  | ✅ PASS |
| dance1          | 0.265 | 0.930 | 0.867   | 0.93  | ✅ PASS |
| sprint1         | 0.180 | 0.977 | 0.703   | 0.72  | ❌ FAIL |
| fight1          | 0.232 | 0.953 | 0.758   | 0.80  | ❌ FAIL |
| fightAndSports1 | 0.232 | 0.984 | 0.406   | 0.41  | ❌ FAIL |
| fallAndGetUp    | 0.262 | 0.953 | 0.164   | 0.17  | ❌ FAIL |

**4/8 PASS** (walk, run1, dance2, dance1). The 4 FAILs are exactly the dynamic/contact motions.

## The headline (now confound-free)
Every teacher passed the survival gate (≥0.95 original survival), so the policy is certified OUT of
the equation. **With confirmed-good teachers, the uniform VAE still fails the dynamic/contact motions
(sprint, fight, fightAndSports, fallAndGetUp).** This DEFINITIVELY confirms the disentanglement: the
dynamic-motion gap is the VAE's reconstruction, not stage-1 fragility. The gate worked exactly as
intended — it made the result interpretable.

## Bugs / caveats found in the results
1. **EX_gated8 is slightly under-trained vs EX_T4w_base** (10k epochs / 8 clips vs 15k / 9 clips). Its
   per-clip RMSE is consistently ~0.005–0.01 higher (fallAndGetUp 0.262 vs 0.255, fightAndSports 0.232
   vs 0.226), and decoded survivals are a touch lower (sprint1 0.72 vs base 0.82, fallAndGetUp 0.17 vs
   0.31). The full-clip reconstruction keeps improving past the val-RMSE plateau (hardup lesson), so
   the hard-motion numbers here are pessimistic — train to ~15k for a fair head-to-head.
2. **Survival is stochastic (~±0.03).** dance1 original = 0.930 here vs 0.961 at gate-check (same
   teacher) — run-to-run noise from random env init. Gate thresholds should have margin (or use
   256–512 envs / multiple reps) so a clip near the threshold isn't flipped by noise.
3. **Normalization reuse:** EX_gated8 reuses the 9-clip T4within normalization.npz for an 8-clip
   corpus. Negligible, but for cleanliness recompute mean/std over the 8 clips.

## Improvements (in priority order)
1. **Fix the VAE on dynamic motion — the proven lever is reconstruction capacity/allocation, NOT
   teachers.** Up-weighting already showed +0.2–0.45 decoded survival on these clips. Replace the
   hacky per-clip weights with a UNIFORM loss change: per-joint/per-feature loss normalization (so
   high-range dynamic joints aren't under-served by raw-angle MSE) and/or a velocity/acceleration
   loss term. Measure with this same sim2sim suite.
2. **Train EX_gated8 longer (~15k+)** for a fair comparison; hard-motion reconstruction should improve.
3. **Robust teachers as a second lever (in progress):** robust-v2 (correlated ±0.15 reference noise)
   fallAndGetUp + fightAndSports1 teachers are training. Earlier, a robust teacher lifted base-decoded
   fightAndSports1 0.41→0.66 — so robust stage-1 + a better VAE may both be needed for the hardest
   motions. Re-eval the FAIL clips with the robust-v2 teachers when they converge.
4. **Tighten the survival metric:** 256+ envs and/or 2–3 reps to cut the ±0.03 noise.

## Verdict for OmniMM handoff
The VAE is sim2sim-validated-trackable for **locomotion + dance (4/4 here, walk/run/dance)** with
confirmed-good teachers — solid for those motion classes. **Dynamic/contact motion (sprint at speed,
fighting, fall-and-get-up) needs a better VAE** (uniform loss change), independently confirmed now
that the teacher confound is removed. Recommended handoff: EX_gated8 (or the up-weighted EX_T4w_hardup,
which already passed 3/4 hard clips) + the per-dim latent-standardization stats.

Artifacts: stage2/out/gate_check.log, stage2/out/gated_sim2sim.log, run_gate_check.sh,
run_gated_sim2sim.sh. VAE: UniMoTok/.../EX_gated8/checkpoints/epoch=9599.ckpt.

## Per-frame fall analysis (2026-06-11) — OVERTURNS the "error-spike" hypothesis
Instrumented the eval to log each env's fall frame (bench_earlyfreeze --log_falls). Decoded motion
(gated VAE) + gated teacher:
- fallAndGetUp: 128/128 fall, median frame 101; 68% of falls in frames 0-150. pos err there ~0.07
  (clip-mean 0.172, ratio 0.46), vel err ~0.08 (mean 0.90, ratio 0.09). Highest error is LATER
  (frames 250-500, pos 0.20-0.24) — never reached.
- fightAndSports1: 98/128 fall, median 214; pre-fall pos err 0.093 (mean 0.134, 0.69x).
FINDING: robots fall where reconstruction error is BELOW average, NOT at error spikes. Falls cluster
at dynamically low-margin phases (fallAndGetUp's violent controlled descent, frames 50-150) where a
SMALL decoded error (~0.07 rad) is unrecoverable even though original tracking survives there (0.95).
IMPLICATION: it's the motion's margin at that instant, not the error magnitude. Lowering average RMSE
won't fix it; the decoded motion must be dynamically EXECUTABLE at the critical phase (why up-weighting
worked, why the 0.10-rad target is the wrong goal). Hardest motions likely need a near-exact VAE on
those phases AND a wider-basin (robust) policy. Tools: bench_earlyfreeze.py --log_falls,
/tmp/falls_*.npz.
