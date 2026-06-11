# Gated stage-1 → uniform VAE → sim2sim — autonomous 8h run (2026-06-10 ~23:40, user asleep)

GOAL (user's clean pipeline): (1) define a SURVIVAL GATE for stage-1; (2) get a gate-passing teacher
for each LAFAN1 motion; (3) SELECT the subset that passes the gate EASILY (good teachers); (4) train
the VAE uniformly on that subset (NO per-clip weights); (5) sim2sim eval → how good is the VAE.
Full autonomy, 4 GPUs (1,2,4,5; 0,3 Blackwell no torch). Robust-policy experiments KILLED (superseded).

## Gate definition
stage-1 ready for a motion = no-reset single-pass ORIGINAL survival >= 0.95 (128 envs, full clip,
LATE thresholds pos>0.25/ori>0.8, via bench_earlyfreeze --no_freeze --late). "Pass EASILY" = reaches
>=0.95 at a MODEST iter count (<=~8k, ~3h) — i.e. converges fast. Hard/slow motions (fallAndGetUp
needs ~12k) are EXCLUDED from the good-teacher subset.

## The 9 LAFAN clips + best existing teacher (model_*.pt)
walk1_subject1            : walk_4090_full30k/29999 (30k)
lafan_run1_subject2       : lafan_run1_subject2_full30k/29999 (30k)
lafan_sprint1_subject2    : sprint1_subj2_full30k/29999 (30k)
lafan_dance1_subject1     : lafan_suite_dance1_subject1/9999 (10k)
lafan_dance2_subject1     : lafan_suite_dance2_subject1/9999 (10k)
lafan_jumps1_subject1     : jumps1_subj1_2h/6000 (6k)
lafan_fight1_subject2     : lafan_suite_fight1_subject2/9999 (10k)
lafan_fightAndSports1_subject1 : fightSports1_subj1_full30k_v2/29999 (30k)
lafan_fallAndGetUp1_subject1   : fallAndGetUp1_subj1_full30k_v2/29999 (30k)
(feature clips live in g1_dataset_T4/{train,test,val}/<clip>.npz; originals in artifacts/<clip>:v0/motion.npz)

## Phases
PHASE 1 (gate-check, ~15min): eval ORIGINAL survival for each clip w/ best teacher → table. Plus an
  early checkpoint (~6k) to record convergence speed. SELECT easy-pass subset.
PHASE 2 (teacher fill, parallel, only if needed): train any motion lacking a passing teacher (most
  already pass at convergence). Use the EARLIEST checkpoint that passes the gate as the teacher.
PHASE 3 (VAE, ~3-4h): build g1_dataset_<subset> (subset clips + recomputed normalization), train a
  uniform VAE (EX_subset, latent128 KL5e-5, ~6-8k epochs is enough — within-val RMSE plateaus early
  but FULL-CLIP keeps improving, judge by sim2sim not val-RMSE). Train from UniMoTok/ w/ .venv_umt.
PHASE 4 (sim2sim, ~45min): decode each subset clip w/ the VAE, track w/ its gated teacher, report
  ORIGINAL vs DECODED survival + ratio. Pair survival with RMSE (survival alone gameable by blandness).
PHASE 5: write stage2/GATED_PIPELINE_RESULTS.md + recommendation.

## Key gotchas (learned)
- Isaac hangs on 2nd gym.make in one process → one tracking pass per process (bench_earlyfreeze).
- Truncate motion to ~800 frames before tracking (gym.make loads whole buffer).
- Stagger Isaac TRAINING launches (USD-generation race → "No contact sensors"); evals less affected.
- Many ckpts/last.ckpt truncated (killed mid-write) → use largest full-size epoch=*.ckpt.
- VAE config has TWO data_dir keys: DATASET.data_dir AND DATASET.params.data_dir (read first) — set BOTH.
- callback.py milestone ckpt save_top_k must be -1 (monitor=None) — already fixed.
- VAE training: ~25-30 epochs/min at 2048; full-clip reconstruction keeps improving past val-RMSE plateau.
- Teacher training: 30k ~ 11h; converges much earlier (walk~6k, fallAndGetUp~12k). 30k is overkill.

## State for restart (if compacted)
Read this file + stage2/out/gate_check.log (Phase 1 table) + GATED_PIPELINE_RESULTS.md.

## EXECUTION STATE (2026-06-11 ~00:20)
Gate-check DONE (stage2/out/gate_check.log): 8/9 pass original survival>=0.95; jumps1 fails (0.914, 6k).
SUBSET = 8 clips (exclude jumps1): walk run1 sprint1 dance1 dance2 fight1 fightAndSports1 fallAndGetUp.
Dataset: stage2/out/g1_dataset_gated8 (T4within subset + reused norm). VAE: EX_gated8, /tmp/cfg_EX_gated8.yaml,
log /tmp/train_gated8.log, ckpts UniMoTok/experiments/biomechanics_tokenizer/EX_gated8/checkpoints/, 10k epochs ~2h.
GPU fleet: 1=VAE EX_gated8; 2=jumps1_gated teacher (recover); 4=fallgetup_rrv2 (robust-v2 teacher);
5=fightsports_rrv2 (robust-v2). Teachers for sim2sim eval = the gate-check teachers (gate_check.log paths).
TODO when VAE done: (1) sim2sim eval all 8 clips w/ gated teachers (decode w/ EX_gated8, track, survival+RMSE);
(2) W&B report for results + UPDATE ROOT/MASTER report (create_root_report.py); (3) analyze bugs/improvements;
(4) ping user. Compare to EX_T4w_base (9-clip) sim2sim. jumps1 teacher (12k) lets us add jumps1 later.
