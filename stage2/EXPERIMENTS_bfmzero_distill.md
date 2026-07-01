# Experiment log — BFM-Zero → control-VAE distillation

Chronological, data-point-driven log of what we tried, what each experiment showed, and why it
points to the next step. Companion to `BFMZERO_DISTILL_PLAN.md` (design) and
`docs/bfmzero_distill_pipeline.html` (offline summary). All on **seed data** (BONES-SEED clips),
Blackwell/4090 GPUs.

---

## Phase 0 — verify BFM-Zero I/O (grounding)

| datapoint | value | why it matters |
|---|---|---|
| action space | **29-D scaled joint-position targets** (`action_scale=0.25` → PD `kp=50`), **not torque** | BC in position-target space is smooth/well-behaved (no torque spikes) |
| teacher latent `z` | **256-D on a √d-sphere** (`project_z`, `norm_z`) | not `N(0,I)` → motivates learning our own Gaussian latent |
| proprio obs `s` | **929-D** dict: `state`(64: dof pos/vel, gravity, ang-vel), `privileged`(`max_local_self`), `last_action`(29), history, time, DR | over-informative — flagged as posterior-collapse risk |
| reference obs `r` | **556 / frame** (`get_backward_observation`) | encoder input |

**Reason forward:** distill `(backward_map, act)` into `MotionVAE`; BC target is the position-target action.

---

## Phase 1 — data collection (`collect_bfmzero_pairs.py`)

Per-step `(s, r, a*, z)` with DR + obs-noise ON and optional DART action noise (step the env with a
noised action, **label with the clean** `a*` → fattens the visited-state tube).

| dataset | clips | DART | purpose | status |
|---|---|---|---|---|
| `bfmpairs_seed40_dart01` | 40 | 0.1 | fast → first sweep | done |
| `bfmpairs_large_dart01` | 569 | 0.1 | full seed set | collecting |
| `bfmpairs_seed569_nodart` | 569 | 0.0 | DART on/off ablation | collecting |
| `bfmpairs_s2k_dart01` | 1962 | 0.1 | scale-up | collecting |

Verified live dims: proprio **929**, ref **556/frame**, action **29**, z **256**.

---

## Phase 2 — offline BC sweep (`train_control_vae.py`, `sweep_control_vae.py`)

72 configs on `seed40`: `H∈{1,8,16,32} × latent∈{16,32,64} × β∈{1e-3,1e-2,1e-1} × align∈{0,0.5}`.

| datapoint | value |
|---|---|
| G1 recon RMSE (val, action units) | **≈ 0.10–0.11** (all configs) — reconstruction is good |
| **G3 z-ablation (max over 72 runs)** | **0.016** — far below the 0.1 gate |
| best-ranked configs | low `β` + `align=0.5` + bigger latent (marginal: 0.016 vs 0.011) |

**FINDING #1 (offline).** The decoder reconstructs the action **from proprio alone**; the latent is
**low-magnitude offline** (near-collapse). No hyperparameter rescues it. → the 929-D proprio largely
determines BFM-Zero's action (a stabilizing controller); the motion-specific residual that `z` should
carry is small and the KL suppresses it.

---

## Phase 3 — closed-loop G2 (`run_control_vae_policy.py`)  ← the decisive gate

Run three policies in the BFM-Zero MuJoCo env, score survival + joint tracking:
`bfm` (teacher) vs `vae_mu` (latent used) vs `vae_zero` (latent ablated).

**2-clip smoke (H32_L16):**

| policy | survival | joint err |
|---|---|---|
| BFM-Zero (teacher) | **1.00** | 11.7° |
| VAE, z=μ | 0.50 | 21.3° |
| VAE, z=0 | 0.00 | 26.0° |

**FINDING #2 (closed-loop — the important one).** Offline BC **does not transfer**: VAE survival
0.50 vs BFM 1.00 (below the 0.9× gate). **Covariate shift is the real blocker**, not latent
structure — the student drifts to states the teacher never labeled.

**FINDING #3 (reconciles #1).** The latent **does matter dynamically**: `z=μ` (0.50) clearly beats
`z=0` (0.00). Even though the offline z-ablation was tiny (0.016), the small per-step action
difference **compounds over the rollout** into survive-vs-fall. → offline z-ablation **understates**
the latent's closed-loop importance; **G2 is the metric that counts**, not G1/G3.

*Running now:* 40-clip G2 across horizons (H1/H16/H32) for solid survival statistics.

---

## Reasoning → next steps (priority order)

1. **The blocker is covariate shift, so the next lever is DAgger, not the latent.**
   Roll out the *student*, relabel student-visited states with `BFM.act(s_student, z)`, aggregate,
   retrain (the faithful Fig-7 loop; `distill_vae.py` already does this vs RL teachers — swap the
   teacher to BFM-Zero). **Expected to lift closed-loop survival toward BFM's 1.0.**
2. **DART-strength ablation (cheap, offline data already being collected).** Train on `dart=0.0` vs
   `dart=0.1` (and later higher) and G2-eval → does state-tube widening alone recover survival, or is
   DAgger required? A clean, controlled datapoint on the covariate-shift axis.
3. **Confirm findings at scale.** Re-run the phase-2 sweep + G2 on the 569 (`large`) and `s2k`
   datasets as they finish — is the 0.10 recon / dynamic-latent picture stable at 10–50× data?
4. **Decoder-input ablation — DEMOTED.** Since the latent already helps dynamically (Finding #3),
   forcing higher *offline* z-ablation is secondary. Revisit only if we later need a cleaner
   generative latent for diffusion.

**Meta-lesson for the pipeline:** G1/G3 (offline recon / z-ablation) are necessary but **misleading
in isolation** — a model can look collapsed offline yet use its latent dynamically, and look good
offline yet fall closed-loop. **Gate on G2 (closed-loop survival) for every design decision.**
