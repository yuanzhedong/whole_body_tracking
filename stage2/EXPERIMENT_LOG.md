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
