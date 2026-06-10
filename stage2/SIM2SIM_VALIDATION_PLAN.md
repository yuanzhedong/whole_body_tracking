# Autonomous sim2sim VAE validation — 2026-06-09 ~18:50, user away ~3h

GOAL: validate the UniMoTok G1-VAE via sim-to-sim (decoded motion still trackable by a
real RL controller) to build the evidence package that convinces Yao He to take the VAE
into the OmniMM diffusion stage. Full GPU autonomy, keep GPUs hot, don't wait for user.

## Method (sim2sim_vae_eval.py)
Phase 0 = offline recon RMSE; Phase 1 = splice decoded joints into the real motion.npz;
Phase 2 = a fully-trained RL tracking policy tracks ORIGINAL vs VAE-DECODED motion in Isaac
(128 envs, NO-RESET survival = fraction never fell). Verdict per clip: decoded survival
>= 0.90x original AND mpbpe <= 1.15x original. Phase 3 = generative readiness (no Isaac).
KEY: sim2sim slowness was CPU contention (see wbt-eval-perf) — Isaac eval must run with the
CPU otherwise idle. So DO ISAAC FIRST (idle), THEN VAE training.

## VAE under test
EX_T4w_base (9-clip LAFAN, latent[1,128]) = UniMoTok/experiments/biomechanics_tokenizer/
EX_T4w_base/checkpoints/last.ckpt. Normalization: g1_dataset_T4within/normalization.npz.
NOTE: many final/last ckpts are truncated (killed mid-write) — use full-size ones.

## Plan
### Phase A — per-clip sim2sim validation (Isaac, GPU 1, SERIAL; one Isaac at a time)
Clips with a competent teacher + full-clip features (T4/{train,test}/<feat>.npz) + artifact:
| tag | feature clip | artifact | teacher ckpt | conf |
|-----|--------------|----------|--------------|------|
| walk    | (T4/train) walk1_subject1        | walk1_subject1        | walk_4090_full30k/model_29999            | HIGH (running 18:40) |
| run1    | lafan_run1_subject2              | lafan_run1_subject2   | lafan_run1_subject2_full30k/model_29999  | HIGH |
| sprint1 | lafan_sprint1_subject2           | lafan_sprint1_subject2| sprint1_subj2_full30k/model_29999        | HIGH |
| dance1  | lafan_dance1_subject1            | lafan_dance1_subject1 | lafan_suite_dance1_subject1/model_9999   | LOW (10k partial) |
Driver: stage2/run_simval_suite.sh (builds temp g1_simval_<tag> = T4within norm + full clip;
runs sim2sim with that clip's teacher; --skip_phase3 --max_steps 400 --eval_reps 1).
Outputs: stage2/out/simval_<tag>.json, logs /tmp/simval_<tag>.log.
The ORIGINAL-tracking survival per clip self-checks teacher competence; if original<0.8 the
teacher is too weak → that clip's decoded result is unreliable (note it, don't over-claim).

### Phase B — keep GPUs hot: higher-KL VAE sweep (GPUs 2,4,5, non-Isaac)
Gen-readiness gate found aggregated posterior = mean 0.65/std 1.68 (NOT N(0,I)); KL 5e-6==5e-5
(both negligible regime). Train T4 VAE at KL in {1e-4, 1e-3, 1e-2} to find the knee where
aggStd→1 without wrecking recon. Run Phase-3 gate (--phase01_only) on checkpoints as they land.
Config via stage2/gen_vae_config.py; train from UniMoTok/ with .venv_umt. Only start AFTER
Phase A's Isaac jobs finish (CPU contention).

### Phase C (stretch) — generative closed-loop
Decode interpolated/prior-sampled latents, splice onto a real root, track → does GENERATED
motion survive? Strongest diffusion-ready evidence. Build only if A+B are solid.

## Deliverable
stage2/SIM2SIM_VALIDATION_RESULTS.md — table of per-clip original vs decoded survival/mpbpe,
Phase-0 RMSE, Phase-3 gen-readiness + standardization stats, and the framing for Yao.

## State / restart
- Walk Isaac run: PID 3534754, log /tmp/simval_walk.log, out stage2/out/simval_walk.json.
- Isaac is a machine-wide singleton → serialize. Idle GPUs: 1,2,4,5 (0,3 Blackwell, no torch).
- If context compacts: read this file + stage2/out/simval_*.json to see what's done; continue.
