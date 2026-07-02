# Overnight Autonomous Session — 2026-06-06 21:20 → ~05:00

User asleep, full GPU autonomy. Goal: understand & break the ~0.36 rad joint-RMSE floor
of the UniMoTok MldVaeBiomechanics VAE (Path A).

## Hypotheses under test

1. **Val-split bug fixed.** Old export put whole clips in val → T1-T3 val=train data
   (meaningless), T4 val=fallAndGetUp=OOD (looked like overfit, was OOD failure).
   FIX: `--split_mode within_clip` holds out last 15% of windows *per clip*.
   Early confirmation: EX_T4w_base val=0.408 (in-dist) << T4_old val=0.468 (OOD).

2. **Is the 0.36 train-RMSE floor capacity or data?** EX_T4w_big (22M, 9-layer,
   latent 256) vs EX_T4w_base (7M, 5-layer, latent 128) on identical T4within data.
   If big breaks the floor → capacity-limited → scale up. If not → data/loss/normalization.

3. **Is KL over-regularizing reconstruction?** EX_T4w_lowkl (KL 5e-6 vs 5e-5 base).

## Running (21:24)

| GPU | job A (old 20k) | job B (new within_clip) |
|-----|-----------------|--------------------------|
| 1 | T1_old + sim2sim Phase-2 | — |
| 2 | T2_old | EX_T4w_base (7M) |
| 4 | T4_old | EX_T4w_big (22M) |
| 5 | T3_old | EX_T4w_lowkl (7M, lowKL) |

## Monitor

`stage2/overnight_monitor.py` (stateless, run per wakeup) → RMSE trajectory for all 7,
logs DONE_* to cs224n-robustqa/g1-vae-ablation on completion. Progress in
/tmp/overnight_progress.md, state in /tmp/overnight_state.json.

## Queued / adaptive decisions (handled on each wake)

- When an old run finishes → launch **EX_T1w_base** (walk-only within_clip) to get the
  honest scaling endpoint vs EX_T4w_base.
- When sim2sim Phase-2 finishes → record first closed-loop gate number; then run sim2sim
  on the best within_clip checkpoint (serialize: Isaac singleton).
- If EX_T4w_big breaks floor (train < 0.30) → queue even-larger / longer.
- If EX_T4w_lowkl helps → queue KL=5e-7.
- Keep all 4 GPUs hot; co-locate ≤2 train jobs/GPU (each ~5-9 GiB, 24 GiB cards).

## Tools built tonight
- `export_g1_motion.py --split_mode within_clip`
- `gen_vae_config.py` (clean config variants → cs224n-robustqa entity)
- `eval_vae_rmse.py` (venv_umt, windowed RMSE)
- `overnight_monitor.py` (orchestrator)
