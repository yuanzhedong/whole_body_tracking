# VAE experiment log — baseline + everything tried (traceable, comparable)

## Fixed eval protocol (so every row is comparable)
- **Clips:** the 8 gated LAFAN clips (walk, run1, sprint1, dance1, dance2, fight1, fightAndSports1, fallAndGetUp).
- **Teachers:** the GATE-PASSING teachers (original survival ≥0.95), FROZEN — same for every VAE.
  Paths in `stage2/out/gate_check.log`. This removes the teacher as a variable.
- **Metric:** no-reset closed-loop **decoded survival** (128 envs, full clip, LATE thresholds), via
  `stage2/run_gated_sim2sim.sh` (swap the `VAE=` line to the variant's checkpoint).
- **PASS** = decoded survival / original survival ≥ 0.90. Survival noise ~±0.03.
- To reproduce any row: edit `VAE=<ckpt>` in run_gated_sim2sim.sh, run, read stage2/out/gated_sim2sim.log.

## BASELINE — EX_gated8
8 gated clips, latent[1,128], KL 5e-5, smooth-L1 recon + velocity(0.5) + root-orient(5) loss,
10k epochs. Dataset `g1_dataset_gated8`. Ckpt `UniMoTok/.../EX_gated8/checkpoints/epoch=9599.ckpt`.

| clip | decoded survival | verdict |
|------|:---:|:---:|
| walk | 1.00 | ✅ |
| run1 | 0.99 | ✅ |
| dance2 | 0.96 | ✅ |
| dance1 | 0.87 | ✅ |
| sprint1 | 0.70 | ❌ |
| fight1 | 0.76 | ❌ |
| fightAndSports1 | 0.41 | ❌ |
| fallAndGetUp | 0.16 | ❌ |
| **PASS** | **4/8** | locomotion+dance pass; dynamic/contact fail |

## VAE experiments (all vs the baseline above)

| variant | delta from baseline | epochs | PASS | hard-clip decoded (sprint/fight/fAS/fall) | status |
|---------|---------------------|:------:|:----:|:------------------------------------------:|--------|
| **EX_gated8** (baseline) | — | 10k | 4/8 | 0.70 / 0.76 / 0.41 / 0.16 | done |
| EX_gated8_dyn | LAMBDA_VELOCITY 0.5→1.0, LAMBDA_ACCELERATION 0→1.0 (dynamics-aware loss) | 10k | — | pending eval | TRAINING (GPU1) |
| EX_gated8_mir | mirror augmentation (2× data, L/R-symmetry; double-mirror verified =0) | 10k | — | pending eval | TRAINING (GPU4) |
| EX_T4w_hardup | hard clips ×4 oversampled (per-clip weight HACK) — proven to help | ~12k | — | 0.91/0.87/0.92/0.69 (w/ full30k teachers, NOT gated-protocol — RE-EVAL pending) | needs re-eval |
| EX_T4w_base | original 9-clip corpus, ~15k epochs | ~15k | — | mixed/old teachers (not protocol) — RE-EVAL pending | superseded |
| EX_T4w_kl{1e-4,1e-3,1e-2} | KL sweep | ~6k | — | n/a (offline only) | NEGATIVE: KL can't normalize latent (aggStd~1.5 regardless) |

Notes:
- EX_T4w_hardup numbers used the full-30k teachers (different protocol) — directionally it fixed 3/4
  hard clips, but RE-EVAL with the gated-protocol teachers for an apples-to-apples row.
- "pending eval" rows: run run_gated_sim2sim.sh with the variant ckpt once training finishes.

## Teacher experiments (stage-1, for the FAIL clips)
| teacher variant | what | fallAndGetUp orig→decoded | fightAndSports1 orig→decoded |
|-----------------|------|:---:|:---:|
| gated baseline (no noise) | standard tracking | 0.95 → 0.16 | 0.98 → 0.41 |
| RobustRef-v1 (±0.3 per-step jitter) | reference obs noise | 0.80 → 0.18 (hurt) | 0.99 → 0.66 (helped) |
| RobustRef-v2 (±0.15 per-episode correlated bias) | reference bias | 0.82 → 0.30 (+0.13) | 0.98 → 0.57 (+0.16) |
→ Robust-v2 helps decoded for both; ±0.15 still slightly too high for precise fallAndGetUp (orig drops).

## Datasets
| name | clips | notes |
|------|-------|-------|
| g1_dataset_T4within | 9 | original within-clip val split |
| g1_dataset_gated8 | 8 | gate-passing subset (excl. jumps1) — BASELINE corpus |
| g1_dataset_gated8_mir | 8×2=16 | gated8 + L/R mirror augmentation |
| g1_dataset_T4within_hardup | 8 (+dups) | hard clips ×4 (up-weighting hack) |

## Key findings driving these experiments
1. RMSE ≠ trackability; survival is the gate (gated pipeline, confound-free).
2. Falls happen at dynamically LOW-MARGIN phases where error is BELOW average — not at error spikes
   (per-frame fall analysis). So average-RMSE reduction may under-deliver; near-exact at critical
   phases is what matters (why up-weighting worked).
3. Hardest motions likely need BOTH a better VAE AND a robust (wider-basin) policy.
4. **BUG in earlier per-joint analysis:** G1 joints are stored INTERLEAVED (left,right,waist,...),
   not L-block/R-block — my earlier "legRMSE/ankRMSE" columns used wrong indices (ignore those
   specific numbers; the avg-over-all-joints fall finding is unaffected). The correct interleaved map
   is now in the mirror code.

## TODO for a fully-comparable table
Re-eval EX_T4w_base + EX_T4w_hardup with the gated-protocol teachers; eval EX_gated8_dyn + _mir when
trained. Then every row uses identical teachers/clips/metric.

## Paper-metric reconstruction (joint-angle RMSE rad; paper biomech target < 0.10)
Computed with stage2/paper_metrics.py (no Isaac). NOTE on setup validation: paper-number REPRODUCTION
is blocked (their AMASS/HumanML3D datasets are absent — /simurgh2 cluster paths gone). Setup confirmed
correct instead via: smoke overfit PASS (synthetic recon 0.236→0.045), reconstruction improving with
data/epochs, and arch/loss/normalization matching the paper config. Our gap to <0.10 = data starvation
(8 clips vs their tens of thousands) + latent[1,128] vs their [1,256], NOT bugs.

| VAE | mean jointRMSE | sprint1 | fightAndSports1 | fallAndGetUp | sim2sim hard-clip survival |
|-----|:----:|:----:|:----:|:----:|:----:|
| EX_T4w_base (9clip) | 0.230 | 0.176 | 0.225 | 0.252 | 0.82/0.41/0.23 (old teachers) |
| EX_gated8 (BASELINE) | 0.233 | 0.179 | 0.230 | 0.256 | 0.70/0.41/0.16 (gated) |
| EX_T4w_hardup (upweight) | **0.197** | **0.131** | **0.182** | **0.183** | 0.91/0.92/0.69 (full30k) |
| EX_gated8_dyn / _mir / _lat256 / _dynmir | pending | | | | training |

KEY RECONCILIATION: across clips, RMSE does NOT predict survival (sprint has lowest RMSE but fails —
motion difficulty varies). But WITHIN a clip, across VAEs, lower RMSE DOES predict higher survival
(hardup drove fallAndGetUp RMSE 0.256→0.183 and survival 0.16→0.69). So the fall-analysis caveat is:
MARGINAL average-RMSE reduction doesn't help, but SUBSTANTIALLY lowering a hard clip's RMSE does.
The hard clips need RMSE pushed well below ~0.20 to start passing — which up-weighting/capacity/data/
dynamics-loss all aim at. Per-group: waist reconstructs best (~0.06, root-orient weight working),
arms worst (~0.25), legs mid (~0.22).

## Paper-metric results for ALL variants (joint RMSE rad) — CAPACITY WINS
| VAE | latent | loss/data delta | mean jointRMSE | hard clips (sprint/fAS/fall) |
|-----|:------:|-----------------|:----:|:----:|
| EX_gated8 (baseline) | [1,128] | — | 0.233 | 0.179/0.230/0.256 |
| EX_gated8_dyn | [1,128] | velocity1.0+accel1.0 | 0.272 (WORSE) | 0.194/0.271/0.326 |
| EX_gated8_mir | [1,128] | mirror 2× data | 0.241 (~neutral) | 0.182/0.239/0.264 |
| EX_gated8_dynmir | [1,128] | dyn+mirror | 0.267 (worse) | 0.195/0.266/0.310 |
| EX_T4w_hardup | [1,128] | hard clips ×4 (HACK) | 0.197 | 0.131/0.182/0.183 |
| **EX_gated8_lat256** | **[1,256]** | **capacity only** | **0.185 (BEST)** | **0.144/0.190/0.192** |

CONCLUSION: latent[1,256] (UniMoTok's own default) is the CLEAN uniform winner — lowest RMSE, beats
the up-weighting hack, no per-clip weights. Our [1,128] bottleneck was limiting. Dynamics loss HURTS
(trades position for velocity/accel accuracy). Mirror is neutral in-corpus (would help generalization
to novel motion, not reconstruction of training clips). Per the within-clip RMSE→survival link,
lat256 should give hardup-like sim2sim survival on hard clips — sim2sim eval pending.
RECOMMENDATION forming: replace the up-weight hack with latent[1,256] + more data; consider [1,512].

## Variant sim2sim survival (decoded, gated teachers) — only CAPACITY helps; data/loss DILUTE at fixed [1,128]
| variant | sprint1 | fight1 | fightAndSports1 | fallAndGetUp | vs baseline |
|---------|:---:|:---:|:---:|:---:|---|
| EX_gated8 baseline | 0.70 | 0.76 | 0.41 | 0.16 | — |
| EX_gated8_dyn | 0.74 | 0.81 | 0.11 | 0.04 | WORSE (dynamics loss hurts) |
| EX_gated8_mir | 0.75 | 0.73 | 0.28 | 0.02 | WORSE (mirror dilutes capacity) |
| EX_gated8_dynmir | 0.72 | 0.88 | 0.28 | 0.02 | WORSE |
| EX_gated8_lat256 | pending (RMSE 0.185 best → expect best survival) |
CRUCIAL INSIGHT: at fixed latent[1,128] (1 token, 128-dim), the VAE is CAPACITY-LIMITED. Adding data
(mirror 2x) or losses (dyn) just splits/reallocates the limited capacity -> WORSE reconstruction of
the originals -> worse survival. INCREASING capacity (lat256) is the only lever that helped.
=> More data MUST come with more capacity, else it dilutes. AMASS experiments must use latent>=256.

## DATA AVAILABILITY (2026-06-11)
- More LAFAN: NONE readily available. HF repo fleaven/Retargeted_AMASS_for_robotics has 0 LAFAN; our
  9 LAFAN clips came from a separate source (/tmp/lafan_suite) not expandable here without raw LAFAN1 BVH.
- AMASS: 17,717 G1-RETARGETED clips available NOW via that HF repo (no manual retargeting!): KIT 4164,
  BioMotionLab 3031, CMU 1978, GRAB 1340, BMLhandball 634, ACCAD 230, MOYO 229, DanceDB 111, SFU 32...
  Pipeline: hf .npy -> retargeting/hf_to_csv.py -> CSV -> csv_to_npz.py -> export_g1_motion.py -> 41-D.
PLAN: pull a dynamic AMASS subset (CMU/ACCAD/DanceDB/handball/MOYO) -> features -> bigger corpus ->
retrain BASELINE(128) + lat256 on SAME corpus (fair) -> eval on the 8 gated clips. Capacity scales with data.

## *** HEADLINE: latent[1,256] (capacity) is the CLEAN fix — replaces the up-weighting hack ***
lat256 full sim2sim (decoded survival, gated teachers) vs baseline[1,128]:
| clip | baseline | lat256 | Δ |
|------|:---:|:---:|:---:|
| walk | 1.00 | 1.00 | — |
| run1 | 0.99 | 1.00 | +0.01 |
| sprint1 | 0.70 | 0.86 | +0.16 |
| dance1 | 0.87 | 0.84 | -0.03 |
| dance2 | 0.96 | 0.98 | +0.02 |
| fight1 | 0.76 | 0.88 | +0.12 |
| fightAndSports1 | 0.41 | 0.91 | +0.50 |
| fallAndGetUp | 0.16 | 0.63 | +0.47 |
PASS: baseline 4/8 -> lat256 ~6-7/8. lat256 RMSE 0.185 (best). This is the CLEAN, uniform replacement
for the up-weighting hack — just UniMoTok's own default latent [1,256] vs our [1,128]; no per-clip
weights, no loss change. Our [1,128] bottleneck was the real limiter. fallAndGetUp (0.63) still <0.90
(hardest motion, but +0.47). RECOMMENDATION: adopt latent[1,256] as the recipe; combine with AMASS
data (+capacity) for the last hard clip. dynamics-loss/mirror at fixed [128] were dead ends (dilution).

## DATA SCALING (raw-root, incremental) — capacity & data are complementary
Full LAFAN1 (40 clips, 2.1h) and AMASS-10h (3042 clips) fetched + converted (raw-root, joints verified).
Corpora: g1_dataset_lafan1 (40), g1_dataset_lafan1_amass (3072 clips, 12.1h). Eval = run_gated_sim2sim_raw.sh.
| VAE | corpus | latent | sprint1 | fight1 | dance1 | fightAndSports1 | fallAndGetUp |
|-----|--------|:------:|:---:|:---:|:---:|:---:|:---:|
| baseline (FK) | LAFAN-9 | 128 | 0.70 | 0.76 | 0.87 | 0.41 | 0.16 |
| lat256 (FK) | LAFAN-9 | 256 | 0.86 | 0.88 | 0.84 | 0.91 | 0.63 |
| EX_lafan1_lat128 | LAFAN1-40 | 128 | 0.97 | 0.96 | 0.70 | 0.36 | 0.13 |
| EX_lafan1_lat256 | LAFAN1-40 | 256 | (training) |
| EX_laA_lat256 | LAFAN1+AMASS 12.1h | 256 | (training, ~5h) STEP 3 |
FINDING: more same-distribution data (LAFAN1) at lat128 HELPS sprint(0.70->0.97)/fight(0.76->0.96)
but DILUTES dance1(0.87->0.70) and doesn't fix the worst (fightAndSports/fallAndGetUp) — not enough
capacity to absorb 40 clips. Capacity[256] alone (LAFAN-9) already lifts all hard clips. So data+capacity
are COMPLEMENTARY: need both. Testing lat256 on LAFAN1 + the 12.1h corpus. fallAndGetUp remains the holdout.

## LAFAN1 @ latent256 (raw) + convention-confound note
EX_lafan1_lat256 (40 LAFAN1, lat256, raw): walk/run1 1.00, sprint1 0.95, dance1 0.875 (dilution FIXED
by capacity), dance2 0.99, fight1 0.94, fightAndSports1 0.48, fallAndGetUp 0.52. 6/8 pass. LOWEST RMSE
yet (walk 0.104, sprint 0.110). Capacity[256] fixed dance1's lat128 dilution -> data+capacity complementary CONFIRMED.
CONFOUND: cross-convention survivals differ at same RMSE (FK lat256 fightAndSports 0.91 vs raw lat256 0.48;
FK fallGetUp 0.63 vs raw 0.52) -> raw-root vs FK-root changes decoded joints via the shared latent. So
only compare WITHIN convention. raw8 baselines (training) give the clean raw-to-raw data comparison.
NEXT: EX_laA_lat512 (12.1h corpus, latent512) launched on GPU1 = max capacity+data.
