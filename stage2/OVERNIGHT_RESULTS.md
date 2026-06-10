# Overnight VAE Ablation — Results Summary (2026-06-06 21:00 → 2026-06-07 ~03:30)

Autonomous session. Goal: understand & try to break the UniMoTok MldVaeBiomechanics
joint-RMSE "floor" and validate the VAE via sim-to-sim. **All experiments use the
UniMoTok `MldVaeBiomechanics` code (Path A), not the custom MotionVAE.**

## TL;DR
1. **The "0.36 rad floor" was a validation-split BUG, not a real ceiling.** Old export put
   whole clips in val → T1-T3 val clips were also in train (meaningless), T4 val was OOD
   (fallAndGetUp). Fixed with `--split_mode within_clip` (hold out last 15% of windows per clip).
2. **Data diversity is the dominant lever.** With honest in-distribution val:

   | clips | train RMSE | val RMSE | gap | verdict |
   |-------|-----------|----------|-----|---------|
   | 1 (walk)            | 0.257 | 0.63–0.66 | ~0.40 | severe overfit |
   | 2 (walk+run)        | 0.31  | 0.47      | 0.16  | improving |
   | 5 (loco+dance)      | 0.36  | 0.42      | 0.06  | generalizes |
   | 9 (all LAFAN)       | 0.38  | 0.40      | 0.04  | generalizes |

   Generalization **saturates by ~5 clips**; val floor settles ~**0.40 rad**.
3. **Model capacity barely helps.** EX_T4w_big (22M, 9-layer, latent 256) vs base (7M, 5-layer,
   latent 128) on identical 9-clip data: at matched epochs (~1500), big = 0.368/0.402 vs
   base 0.380/0.412 — a **~0.01 rad edge, within noise**. 3× params is NOT a path to 0.10.
4. **KL weight: null.** 5e-6 vs 5e-5 → no meaningful difference in reconstruction.
5. **sim2sim Phase-2 (closed-loop): NOT OBTAINED — blocked by resource contention.** Ran 3 attempts
   (full clip, fast eval_reps=1, bounded short clip); ALL CPU-starved by the 9-10 co-located VAE
   training jobs (each spawns DataLoader workers saturating the 32 cores). The Isaac env-step loop
   crawled at ~1s/step (vs ~150ms normal). **Phase-2 MUST run on a dedicated idle machine** (no
   concurrent training). This is the #1 unfinished item — run it once the VAE fleet stops.

## Interpretation
The 0.10 rad gate is ~4× below the achieved val floor (~0.40 rad). **More LAFAN clips will NOT
close it** (already saturated), and **naive capacity scaling will NOT close it** (~0.01 rad gain).
Reaching 0.10 — if it's the right target — likely requires a different lever:
- different motion representation (e.g. per-joint normalization, residual/delta encoding),
- stronger/longer-context decoder or larger latent *with* much more diverse data,
- per-clip/per-frame quality filtering (remove bad retargeted frames that cap reconstruction),
- or revisiting whether 0.10 rad is the right benchmark for this VAE+representation
  (it may come from a different data/feature setup in the original UniMoTok work).

## Recommended next steps
1. **Run a clean sim2sim Phase-2 on a DEDICATED idle GPU** (no co-located training) to get the
   closed-loop survival/mpbpe gate. Use a short clip + eval_reps=1. The number tells you whether
   ~0.40 rad val even matters for tracking (the policy may be robust to it).
2. **Don't invest in more LAFAN clips for the VAE** — saturated. If pushing data, add *categorically
   new* motion (AMASS diversity) rather than more of the same.
3. **Try representation/loss changes** before more scale: per-joint std normalization, longer windows
   (256), or an L2+velocity loss reweighting.
4. **Decide on the 0.10 target** — sanity-check it against the original UniMoTok biomechanics numbers.

## Artifacts & where to look
- **W&B:** `cs224n-robustqa/g1-vae-ablation` — runs EX_T1w_base, EX_T2w_base, EX_T3w_base,
  EX_T4w_base, EX_T4w_big, EX_T4w_lowkl (+ DONE_*/final_* summary runs). Loss curves under each run.
- **Live RMSE monitor:** `stage2/overnight_monitor.py` (stateless, re-run anytime).
  State `/tmp/overnight_state.json`, cycle log `/tmp/overnight_progress.md`.
- **Tools built:** `gen_vae_config.py` (config variants), `eval_vae_rmse.py` (windowed RMSE),
  `export_g1_motion.py --split_mode within_clip` (the val-bug fix), `overnight_monitor.py`.
- **Datasets:** `stage2/out/g1_dataset_T{1,2,3,4}within` (proper in-distribution val splits).
- **Note:** the 20k-epoch EX runs were NOT finished by morning (~ep5800/20000 for base after 6h;
  big much slower). They're still training or can be stopped — the scaling/capacity conclusions
  hold from the trajectories. The old T1-T4 20k runs (flawed val) finished/finishing; ignore their
  val numbers, only their training-loss convergence is meaningful.

## Methodology caveats (so you trust the numbers)
- within_clip val shares ~127 frames of window-context with train (window-1 overlap); minor.
- RMSE is windowed (128-frame, stride 64, overlap-averaged) over joint dims 12:41, in radians.
- All RMSE computed with the VAE's TRAINING normalization stats (critical — wrong norm = garbage).

## sim2sim Phase-2 — RESOLVED qualitatively (2026-06-08, dedicated GPU)
Freed disk + killed fleet, ran sim2sim on a DEDICATED GPU (no contention). Confirmed starvation
diagnosis: Phase 0+1 = 2.5 min un-starved vs 80 min+ starved. BUT the decoded-reference Phase-2
run never completes in reasonable wall-clock (killed at 23 min on the 33s short clip): the policy
tracking the VAE-decoded walk (~0.35 rad mean joint error ~= 20 deg/joint) FALLS almost every step,
so nearly every sim step becomes a 128-env reset -> pathologically slow AND a clear verdict.

CONCLUSION: ~0.35 rad reconstruction error BREAKS closed-loop tracking. Decoded survival ~= 0;
Phase-2 gate (decoded survival >= 0.90x original) FAILS by a wide margin. So RMSE quality DOES
matter -- the current VAE (val floor ~0.40 rad) is not good enough to feed the tracker. Reaching
near the ~0.10 target is genuinely needed for generate-then-track to work.

TOOLING FIX NEEDED: sim2sim Phase-2 must add early-termination/step-cap so a failing decoded run
ends fast with survival~0 instead of thrashing 20+ min. Never co-locate Phase-2 with training (CPU starve).

## sim2sim Phase-2 — CORRECTION (2026-06-08, after early-termination fix)
EARLIER CLAIM ("0.35 rad breaks tracking") was PREMATURE — it was based on wall-clock thrashing,
not failure counts. Implemented early-termination in sim2sim_vae_eval.py (--max_steps, --fail_mult,
--min_steps): a constantly-FALLING run trips "cumulative failures > num_envs*fail_mult" within ~200
steps and bails fast with survival~0; the loop also records n_terminated / n_completed / early_exit.

OBSERVED across multiple dedicated-GPU runs (~2500 decoded steps total): the failure early-exit
NEVER FIRED. If the robot were toppling on the decoded motion it would have bailed in ~200 steps.
It didn't — so the robot STAYS UPRIGHT. The decoded-run slowness is from RSI clip-completion resets
(envs finishing the short clip and restarting), NOT from falls.

REVISED CONCLUSION (evidence-based; exact survival_ratio still pending a faster eval):
~0.35 rad joint-reconstruction error LIKELY DOES NOT break closed-loop tracking. Phase 1 keeps the
ORIGINAL root trajectory and only swaps decoded joint reference poses; the policy tracks the correct
root and holds balance via proprioception, so degraded joint references raise pose error (mpjpe) but
don't topple the robot. ENCOURAGING for generate-then-track: the ~0.40 rad VAE val floor may be fine
for trackability even if not ideal for motion fidelity. (This REVERSES the earlier overcall.)

CAVEAT: exact survival_ratio NOT measured — eval too slow on this shared box (~0.7-1.5 s/step:
128 envs, GPU->CPU .item() sync every step, single-thread Python loop, contended CPU). To get the
number: idle machine + more envs + one clip pass + metrics synced once at end (not per-step), or a
vectorized engine (MJX). Use the n_terminated/n_completed counts, NOT wall-clock, as the signal.

## ROOT CAUSE of slow eval — MEASURED (2026-06-09, micro-benchmarks)
Two benchmarks: stage2/bench_decompose.py (per-step cost breakdown) + stage2/bench_reset.py
(orig vs decoded motion). Findings:

PER-STEP DECOMPOSITION (128 envs, walk_0_33, GPU @43%):
  policy inference   0.16 ms   (negligible)
  .item() readout    0.08 ms   (NEGLIGIBLE — the per-step GPU->CPU sync hypothesis is FALSE)
  env.step (synced) 34.7  ms   (~all of it)
  env.step (no-sync)30.5  ms   (~same -> no GPU pipeline slack to reclaim)

RESET BENCHMARK (300 steps, 128 envs, same machine/load back-to-back):
  ORIGINAL motion:  39.5 ms/step, 0 resets, 0 terminations (robot tracks perfectly) -> 310 steps in ~12s
  DECODED motion:  ~2000+ ms/step (~50-65x slower); robot FALLS -> resets every step -> 300 steps took 13+ min

ROOT CAUSE: it is NOT the .item() sync, NOT policy, NOT few envs per se. Two layers:
1. Clean env.step() for 128 envs is ~40 ms — FINE (~1 min for a full clip). On trackable motion the eval is fast.
2. The blow-up is RESET-ON-FALL OVERHEAD. When the reference motion is untrackable (VAE-decoded), the robot
   terminates (falls) almost every step; Isaac's reset (re-init robot state + write to PhysX) is far costlier
   than a clean step, giving ~50-65x slowdown. THAT is why closed-loop eval on decoded motion crawled.
(The overnight slowness was a SEPARATE issue: CPU starvation from 10 co-located training jobs.)

CORRECTION TO EARLIER CLAIM: the decoded run is slow BECAUSE THE ROBOT FALLS. So my "early-exit never fired ->
robot survives" inference was WRONG (the fail-budget threshold was just set too high to trip in short runs).
The VAE-decoded motion is NOT trackable; ~0.35-0.40 rad joint error DOES break tracking. VAE quality genuinely
matters and the ~0.10 rad target is meaningful, not arbitrary.

IMPLICATIONS for getting a clean Phase-2 number fast:
- The early-termination fix (--fail_mult) is the RIGHT approach but needs a LOWER threshold (e.g. fail_mult=1-2,
  min_steps=80) so a falling run bails in <100 steps instead of grinding through resets.
- Better: terminate the whole run once survival is statistically decided; or measure survival over ONE clip pass
  with envs that DON'T auto-reset (so a fall ends that env cleanly instead of resetting+continuing).
- Clean eval on GOOD motion is fast (~40ms/step) — slowness is specific to evaluating BAD (untrackable) motion.

## mjlab/MJX reset-cost — CONFIRMED by benchmark (2026-06-09, stage2/bench_mjx_reset.py)
Tested the architectural claim directly: in MJX (GPU MuJoCo, what mjlab uses) is RESET as cheap as
STEP? Benchmark: batched humanoid (nq=28), jitted step vs jitted step+reset-ALL-envs-every-step.
Env .venv_mjx: mujoco 3.9 + mujoco-mjx + jax[cuda12] 0.6.2, GPU 4.

    envs |  step only | step+reset-all | reset overhead
     128 |   0.361 ms |     0.336 ms   |   ~0   (noise)
    1024 |   0.606 ms |     0.599 ms   |   ~0   (noise)
    4096 |   1.236 ms |     1.347 ms   |  +0.11 ms (~9%)

CONFIRMED: resetting EVERY env EVERY step in MJX adds ~0 cost (<=9% even at 4096 envs). Reset is a
vectorized where-select over the flat qpos/qvel state, FUSED into the same jitted GPU kernel as the
step. A constantly-falling robot costs the same as a perfectly-tracking one.

CONTRAST (measured earlier, Isaac/PhysX, G1, 128 envs): clean step ~40 ms; step WITH reset-on-fall
~2000 ms -> ~50x blowup. So the reset-on-fall blowup that makes Isaac eval crawl on untrackable
motion DOES NOT EXIST in MJX/mjlab. Bonus: MJX step itself ~0.36 ms vs Isaac ~40 ms (~100x faster
per step) -- caveats: generic MuJoCo humanoid (not G1), no USD/render layer. Engine-level mechanism
confirmed; a full G1 tracking-task port would be the end-to-end word, but the reset/step asymmetry
(the root cause) is settled: MJX reset = free, Isaac reset = ~50x.

## Isaac-native eval speedup — implemented + honest verdict (2026-06-09)
Goal: make sim2sim Phase-2 fast WITHOUT leaving Isaac. Root cause was reset-on-fall (~50x).
Implemented in sim2sim_vae_eval.py (all in the default --no_reset path):
1. NO-RESET: disable fall-terminations so a fallen robot is NOT reset (avoids the ~50x reset).
   survival = fraction of envs that never fell (detected via the same bad_anchor_* fns). WORKS.
2. SURVIVAL EARLY-EXIT: bail once >=95% of envs have ever-failed (survival<=5% locked in). Fast
   for the total-failure case. WORKS.
3. MAX_STEPS CAP (default 500 = 10s @50Hz): a fall is decided in <~2s; surviving 10s ~= surviving
   the clip. Bounds runtime. This is the REAL general speed fix. WORKS.
4. FREEZE-FAILED (--freeze_failed, default OFF): tried parking fallen robots high each step. BACKFIRED
   — write_root_state_to_sim re-syncs PhysX every step (~as costly as a reset). Left off by default;
   a park-ONCE-then-set-kinematic version would be the right approach (future work).

HONEST VERDICT: eval now COMPLETES reliably (~3-4 min vs 20+ min thrashing). But it is NOT uniform
~40ms/step: clean stepping (upright robot) is ~40ms; a fallen ragdoll is ~240-420ms/step due to
contact-solver cost on the collapsed body. The max_steps cap bounds this; it doesn't eliminate it.
Truly uniform-fast eval needs either a working kinematic-freeze of failed envs, or MJX/mjlab (where
reset is free AND step is ~100x cheaper — benchmarked above). For now: no-reset + early-exit +
max_steps=500 is the practical Isaac-native config and is good enough to unblock VAE validation.

## ROOT CAUSE — CORRECTED AGAIN (2026-06-09, validated before implementing freeze)
Quick validation benchmarks (stage2/bench_kinfreeze.py + a policy-on-orig-vs-decoded test) OVERTURNED
the "fallen-ragdoll contact pile" explanation. Measured per-step time, 128 envs, walk_0_33:
  - policy tracking ORIGINAL motion (robot tracks):        ~23 ms/step  (FAST)
  - fallen robot, ZERO action (limp, settled on ground):   ~36 ms/step  (also cheap!)
  - policy tracking DECODED motion (robot fails):          ~2000-3000 ms/step  (~100x, CATASTROPHIC)

=> The slow cause is POLICY-DRIVEN THRASHING, not passive contacts. A *settled* fallen robot is cheap
   (36ms). But when the policy ACTIVELY drives a robot to track motion it can't follow, the robot
   scrambles violently (high joint velocities, hard high-speed contacts, fighting the fall) — that
   churn is what the contact solver chokes on (~2-3 s/step).

IMPLICATION for the fix: KINEMATIC-FREEZE is still the right fix but for a DIFFERENT reason than first
stated — a failed robot is slow because the POLICY KEEPS DRIVING IT. Freezing it kinematic (joint
commands ignored, body not dynamically simulated) STOPS the thrashing -> that env goes cheap. This
also explains why my per-step write_root_state "freeze" failed: the robot stayed dynamic AND
policy-driven (still thrashing) and paid a re-sync cost on top.

Prior "ragdoll contact pile" notes above are SUPERSEDED by this measured result.

## FINAL VERDICT on Isaac eval speed (2026-06-09) — I hit a ceiling, did NOT solve it
Measured root cause is solid: POLICY-DRIVEN THRASHING on unfollowable motion (~2000ms/step) — a
failing robot is violently driven to chase impossible reference poses -> high-speed churning
contacts the PhysX solver chokes on. (NOT the .item() sync, NOT reset alone, NOT a passive ragdoll
pile — a SETTLED limp robot is ~36ms.)

Tried, in order, each helping partially then hitting a new floor:
  - no-reset (disable fall-terminations)        -> removed the ~50x reset-on-fall
  - survival early-exit (bail at >=95% failed)   -> fast ONLY when ~everything fails
  - max_steps cap                                -> bounds step COUNT, not per-step cost
  - park-failed-high (write_root every step)     -> BACKFIRED (per-step PhysX re-sync)
  - action-zero failed envs                      -> ~2.5x (PD still springs to default pose)
  - stiffness=0 limp (ImplicitActuator)          -> ~3-4x but per-step write .cpu() syncs + settling
                                                    contact churn keep decoded at ~hundreds ms-2s/step

OUTCOME: even a 120-step decoded run would not finish in <5 min. I did NOT get a clean DECODED
survival number, and I do NOT have a fast Isaac eval. ORIGINAL survival=1.0 is the only solid
Phase-2 datapoint (robot tracks real motion perfectly, ~40ms/step, fast).

WHY freeze underdelivered: (1) write_joint_stiffness/velocity_to_sim do a full-buffer GPU->CPU
transfer + sync EACH call, fired every step during the ~100-step failing window -> adds sync cost
that offsets the gain; (2) 128 robots SETTLING (even limp) churn contacts for many steps; my "36ms"
was FULLY-settled robots, which the eval never reaches.

THE ACTUAL FIXES (not done):
  A. mjlab/MJX — benchmarked: free resets, ~100x cheaper steps, no per-step PhysX re-sync. The clean
     answer. See FUTURE_WORK_mjlab_port.md.
  B. A sync-free Isaac freeze: set stiffness=0 via a GPU-resident write (no .cpu()), OR truly make
     the failed articulation kinematic once. Possible but more PhysX-API work.

PRACTICAL RECOMMENDATION: use Phase-0 RMSE (fast, no Isaac) as the routine VAE-quality signal.
Treat Phase-2 closed-loop as an expensive spot-check (minutes/clip, only on GOOD-looking VAEs where
the robot won't fall). For Phase-2 at scale, go to mjlab.

## CORRECTION (2026-06-09 late) — the slowness was CPU CONTENTION, not thrashing/resets/PhysX
Re-ran the Phase-2 step-cost benchmark on a now-IDLE box (no co-located training) with
`bench_earlyfreeze.py` (128 envs, decoded clips, windowed ms/step). Result OVERTURNS the
"policy-driven thrashing" verdict above:

| case (idle box, 128 envs)                    | survival | ms/step |
|----------------------------------------------|----------|---------|
| good-VAE walk (et_decoded)                    | 0.992    | ~24     |
| bad-VAE dance, NO resets                       | 0.359    | ~28     |
| bad-VAE dance, resets ON                       | 0.742    | ~28     |
| bad-VAE fallAndGetUp, resets ON (TOTAL fail)   | 0.000    | ~32     |

EVEN the worst case — 128/128 envs fallen, resets firing every step — holds ~32 ms/step. So:
- The ~2000-3000 ms/step "catastrophe" was **NOT** reset-on-fall, **NOT** policy-driven thrashing,
  **NOT** a PhysX contact-solver limit. All of those run ~24-34 ms/step on an idle machine.
- The ONLY variable that differed when 2000 ms/step was measured: **9-10 co-located VAE training
  jobs saturating all 32 CPU cores** (the old notes flagged this as "~1 s/step vs ~150 ms normal"
  but then mis-attributed the floor to PhysX). Isaac env-step is CPU-bound on its python/PhysX host
  side; starving the cores stretches every step ~50-100x.

CONSEQUENCES:
- The early-freeze / kinematic-freeze / stiffness=0-limp work and the mjlab/MJX-port recommendation
  were all chasing a contention artifact. Not needed for a fast Isaac eval.
- OPERATIONAL RULE: never co-locate sim2sim Phase-2 with training. On a free box it is already
  ~28 ms/step (~6 s for a 200-step pass, 128 envs) regardless of survival.
- The survival metric still works fine via the bad_anchor_* detectors with terminations DISABLED
  (no-reset) — but disabling resets is a convenience, not a speed requirement.
