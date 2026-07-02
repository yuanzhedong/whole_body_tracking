# Autonomous 4h plan (2026-06-11 ~08:30, user away 4h)

User asks: keep GPUs hot; reproduce UniMoTok paper numbers to confirm correct setup; run UniMoTok on
our data + data found necessary; report PAPER METRICS (joint-angle RMSE rad / MPJPE) AND sim2sim
success rate; any other useful work. Full autonomy, no confirmation.

## BLOCKER: paper-number reproduction
UniMoTok paper datasets ABSENT (/simurgh2/datasets/AMASS_SMPLH_Bio_20fps, HumanML3D, bioamass_v1.0 —
all gone; no local sample). Can't reproduce their exact numbers. Validate setup instead via:
(a) smoke overfit (synthetic batch -> low recon = arch/loss correct);
(b) our-data reconstruction improves with data/epochs (pipeline works);
(c) config/arch/loss match paper (normalize_motion, smooth-L1, velocity+root-orient loss, skip-transformer).
NOTE: their <0.1 rad is on huge data + latent[1,256]; ours is latent[1,128] on 8 clips -> gap is data
+ capacity, not bugs. Document this honestly.

## Paper metric = joint-angle RMSE (rad), target <0.1 (their biomech headline). We already compute it
(sim2sim Phase-0). Add per-joint-GROUP RMSE with CORRECT INTERLEAVED indices (legs/arms/ankles).
Joint order is interleaved (left,right,waist,...) NOT L-block — see stage2/mirror_augment.py NAMES.
Optional: MPJPE(mm) via FK (needs URDF kinematics; defer unless quick).

## Eval EVERY VAE with BOTH metrics (fixed protocol = gated teachers, 8 clips):
paper joint-RMSE (rad, CPU, no Isaac) + sim2sim decoded survival (Isaac). Fill EXPERIMENT_LOG.md table.
VAEs: EX_gated8 (baseline), EX_gated8_dyn, EX_gated8_mir, EX_T4w_hardup, EX_T4w_base. Re-eval
hardup/base with gated teachers for comparability.

## Keep GPUs hot — variant queue (free GPUs: 1,4 when dyn/mir finish ~2h; 2,5 other users)
RUNNING: EX_gated8_dyn (vel+accel loss, GPU1), EX_gated8_mir (2x mirror data, GPU4).
QUEUE (launch as GPUs free): EX_gated8_lat256 (latent[1,256] = UniMoTok default capacity — tests if
128 is limiting + closer to their setup); EX_gated8_dynmir (dyn+mirror combo); EX_gated9 (add recovered
jumps1 -> 9 clips); EX_gated8_legw (leg/ankle joint loss weighting, interleaved-correct).
Config recipe: clone EX_T4w_base config, set BOTH DATASET.data_dir + DATASET.params.data_dir, END_EPOCH
10k, train from UniMoTok w/ .venv_umt. callback.py save_top_k=-1 already fixed.

## More data
Mirror aug = free 2x (EX_gated8_mir, done). jumps1 now has a gated teacher (14k, surv 1.000) -> 9-clip
corpus available. More LAFAN/AMASS = retargeting (heavy) -> recommend, likely defer in 4h.

## Report
Update EXPERIMENT_LOG.md (both-metrics table) + W&B report + root. Document paper-repro blocker.

## State / restart
EXPERIMENT_LOG.md = comparable table. gate_check.log = gated teachers. run_gated_sim2sim.sh = sim2sim
(swap VAE=). Smoke: UniMoTok/smoke_mld_vae.py. GPUs: nvidia-smi; my trainings = grep run_name/EX_gated.
