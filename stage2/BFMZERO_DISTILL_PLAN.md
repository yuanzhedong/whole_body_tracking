# Distilling BFM-Zero into a control VAE (reference → action)

**Goal.** Learn a compact conditional VAE that maps **reference motion → humanoid action**, distilled
from **BFM-Zero** (the Forward-Backward foundation-model tracker) as the teacher, with an `N(0,I)`
latent, trained on **(state, reference, action)** triples collected under **domain randomization**,
and validated **sim2sim** in MuJoCo. This is the faithful BeyondMimic **Fig-7 control VAE**, but with
BFM-Zero as the single generalist teacher instead of per-clip RL policies.

See also: `stage2/unimotok_vs_figure7.md` (the motion-VAE vs control-VAE analysis),
`stage2/vae_model.py` (the control-VAE we reuse), `stage2/distill_vae.py` (the DAgger loop),
`stage2/verify_vae.py` (the G1–G4 gates), `docs/PIPELINE_SETUP.md` (the generate-then-track pipeline
this complements).

---

## 1. Why distill BFM-Zero (vs the validated generate-then-track pipeline)

The current pipeline is two stages: motion-VAE **reconstructs** a clip → a **separate** tracker
(BFM-Zero / HoloMotion) executes it. This plan **fuses** them into one latent→action controller.
Payoffs:

1. **One deployable policy.** `a = D(z, proprio)` replaces "decode motion, then track it."
2. **A diffusion-ready latent.** BFM-Zero's `z` lives on a `sqrt(d)`-sphere (`project_z`, `norm_z`),
   not a Gaussian. The KL term gives an `N(0,I)` action-latent — what OmniMM Phase-3 diffusion needs
   (raw actions have torque spikes that break diffusion; the VAE smooths them). See
   [[project-omnimm-handoff]].
3. **Inherit BFM-Zero coverage** — including near-ground crouch/sit/squat that HoloMotion collapses
   on — in a small student.

## 2. The key structural fact (verified in the BFM-Zero source)

BFM-Zero already factorizes exactly like a control VAE
(`humanoidverse/agents/fb/model.py`, `stage3_sim2sim/bfmzero_compare/batch_tracking_inference.py`):

```python
z = model.backward_map(ref_obs)      # reference/goal motion -> latent z   (E)
a = model.act(proprio_obs, z)        # z + current state     -> action     (D)
```

So we are **distilling the pair `(backward_map, act)` into a VAE with a Gaussian bottleneck**, not
inventing a controller. Verified interface:

| symbol | source | shape / semantics |
|---|---|---|
| `r` reference obs | `get_backward_observation(env, mid)` → "humanoid_observations_max" | per-frame ref body pos/rot/vel/ang-vel, local-root |
| `s` proprio obs | `wrapped_env._get_g1env_observation()` | actor input (joint pos/vel, base, last action, …) |
| `z` teacher latent | `project_z(backward_map(r))` | on `sqrt(d)`-sphere, **not** `N(0,I)` |
| `a*` target action | `model.act(s, z, mean=True)` | **29-D scaled joint-position targets** (offset from default pose; `action_scale=0.25` → PD `kp=50`), **not torque** |

**Implication:** BC in the 29-D **position-target** space is well-behaved (no torque spikes) — a good
reconstruction target.

## 3. Model (reuse `stage2/vae_model.py`)

`MotionVAE(ref_dim=|r|, proprio_dim=|s|, act_dim=29, latent=32)`:
- **Encoder** `E(r) → (μ, logσ)`, `z ~ N(μ,σ)` — *our* smooth latent.
- **Decoder** `D(z, s) → â` — **must** be state-conditioned (under DR/noise the same reference needs
  different corrections in different states; motion→action without state fails closed-loop).
- **Loss** `‖â − a*‖² + β·KL(q(z)‖N(0,I))` (+ optional latent-align `‖z − project(backward_map(r))‖²`
  so the Gaussian latent stays semantically tied to BFM-Zero's `z`).

**Encoder horizon `H` (first-class hyperparameter).** The encoder sees an `H`-frame reference window
→ `z`, `z` is held over the horizon, and the decoder emits an action per step with proprio. `H` is a
**sweepable knob**, not a fixed choice:

| `H` | regime | notes |
|---|---|---|
| `1` | per-step causal (Fig-7 exact) | most reactive; latent = instantaneous intent |
| `~8–16` | short horizon (**recommended default**) | near-term intent; `16` lines up with the Fig-7 diffusion horizon; diffusion-friendly, closed-loop stable |
| `≥64` | long / whole-clip (UniMoTok MLD-VAE regime) | good for *generation*, **too long for control** — latent gets diffuse, the Gaussian bottleneck strains, action over-smooths |

**Do not confuse this with the motion-reconstruction VAE's 128-frame window** (`config_g1_seed_512_fixed.yaml`,
`window_size: 128` ≈ 6.4 s @ 20 fps). That length is *appropriate for generation* (compress a whole
clip → one global token). For **control**, a single action depends on the current state + near-term
reference, so `H` should be **short** (per-step to ~16). Short for control, long for generation.

**Where `H` lives:** it is a **training-time** parameter (phase 2), **not** a collection parameter.
`collect_bfmzero_pairs.py` logs the **full per-step reference sequence per motion**, so any `H` window
(past or future frames) is a cheap train-time slice — sweep `H` without re-collecting. Pre-windowing
in the collector would bake in one `H` and inflate storage ~`H×`; per-step logging is strictly more
flexible.

## 4. The hard problem: covariate shift, and where DR / DART fit

The student **is** the tracker now, so its own small errors feed back — naive offline BC drifts to
states BFM-Zero never visited → compounding error → fall. Three escalating fixes:

1. **DART / noise-injection BC (start here).** During collection, apply a **noised** action
   `a_env = a* + ε` to *step* the env (fattening the visited-state tube), but **label** state `s_t`
   with the **clean** teacher action `a*(s_t)`. Purely offline; no teacher in the train loop.
2. **DAgger (faithful Fig-7).** Roll out the *student*, query `BFM.act(s_student, z)` at
   student-visited states, aggregate, retrain. Kills covariate shift; needs BFM-Zero live in the
   loop. `stage2/distill_vae.py` already implements this against RL teachers — swap the teacher.
3. **On-policy + OU noise + DR** (the full Phase-2 recipe) if 1–2 are insufficient.

**Domain randomization has two distinct roles — don't conflate:**
- **Robustness (sim2sim/sim2real):** randomize mass/friction/motor-strength/latency + obs noise so
  the action is robust. **Already exposed** in the env config
  (`disable_domain_randomization=False`, `disable_obs_noise=False`).
- **Covariate coverage:** DR also perturbs the state distribution → partial substitute for DAgger.
- **Subtlety:** BFM-Zero was trained *with* DR, so it emits DR-robust actions; distilling under DR
  reproduces that. **Match BFM-Zero's DR ranges first, then widen** — if BFM-Zero degrades under your
  ranges, the student inherits that ceiling.

## 5. Phased plan (maximal reuse)

| phase | deliverable | reuse | status |
|---|---|---|---|
| **0** confirm BFM-Zero I/O | action = 29-D pos-targets; `act`/`backward_map`/`project_z` signatures; z-dim | `humanoidverse/agents/fb/model.py` | ✅ done (§2) |
| **1** data collector | `stage2/collect_bfmzero_pairs.py`: per-step `(s, r, a*, z)` with DR + DART noise → npz | `batch_tracking_inference.py` | **this PR (prototype)** |
| **2** offline VAE-BC | `stage2/train_control_vae.py`: train `MotionVAE` on triples (**sweep encoder horizon `H`**, §3); reports G1 recon RMSE + G3 active-dims/z-ablation | `vae_model.py` | **this PR (prototype, smoke-tested)** |
| **3** closed-loop gate | run distilled VAE **policy** in MuJoCo; survival vs BFM-Zero (**hard gate G2**) | `verify_vae.py` G2, `stage3_sim2sim/` | next |
| **4** DAgger (if G2 red) | student-rollout + BFM-Zero relabel in humanoidverse | `distill_vae.py` (teacher→BFM-Zero) | conditional |
| **5** sim2sim | HV/Isaac-trained VAE → MuJoCo across DR seeds; survival/tracking vs BFM-Zero on near-ground suite | `run_l3_eval.py` pattern | next |

## 6. Verification gates (from `verify_vae.py`, teacher = BFM-Zero)

| gate | proves | pass |
|---|---|---|
| **G1** reconstruction | decoder learned BFM-Zero's map | `‖â − a*‖²` small |
| **G2** closed-loop *(hard)* | the VAE *policy* controls the robot | survival ≥ 0.9× BFM-Zero, E_mpjpe ≤ 1.3× |
| **G3** latent structure | latent smooth, used, ≈`N(0,I)` | z-ablation ≥ 0.1, ≥2 active dims |
| **G4** robustness | recovery basin under DR/OU noise | recovers |

**Do not proceed to diffusion until G2 + G3 pass.**

## 7. Risks / open decisions

- **State-reactivity vs bottleneck:** if BFM-Zero actions are highly state-reactive, a 32-D Gaussian
  bottleneck may not reconstruct them without a bigger latent or the proprio-decoder doing most of
  the work — then the "VAE" is a conditional policy whose latent carries only coarse intent. Watch
  z-ablation early. (Latent target: **free Gaussian** vs **align to BFM `z`** — lean align.)
- **DR-range matching** (§4 subtlety) — mismatch caps the student.
- **Reference featurization parity:** our encoder input `r` should be BFM-Zero's backward obs (reuse
  `get_backward_observation`) so the encoder sees what produced `z`.
- **Env cost:** collection + DAgger need the BFM-Zero humanoidverse env (GPU + MuJoCo/Isaac); offline
  BC training runs in the light `.venv6`-style env.

## 8. Reproduce (phase 1)

Run in the **BFM-Zero env** (imports `humanoidverse`):

```bash
python stage2/collect_bfmzero_pairs.py \
    --model-folder <bfm_zero_ckpt_dir> \
    --data-path   <motions.pkl> \
    --out-dir     <dataset_dir> \
    --dart-noise-std 0.1 \      # 0 = pure BC; >0 = DART state-tube
    --domain-rand --obs-noise \
    --max-steps 1200
```

Emits `pairs_<mid>.npz` with `proprio[T,ds]`, `ref[T,dr]`, `action[T,29]`, `z[T,dz]` (+ meta), the
`(s, r, a*)` dataset for phase 2.

**Phase 2** (plain torch env, e.g. `.venv6`; no simulator):

```bash
.venv6/bin/python stage2/train_control_vae.py \
    --data-dir <dataset_dir> --out stage2/out/control_vae \
    --horizon 16 --latent 32 --beta 0.01 --align-coef 0.0 \
    --epochs 50 --batch-size 1024 --lr 5e-4 --val-frac 0.1
# sweep --horizon {1,8,16,32}; watch G3 z-ablation (>=0.1) and G1 recon RMSE.
```

Writes `control_vae.pt`, `normalization.npz`, `metrics.json` (G1 recon RMSE, G3 active dims +
z-ablation). Closed-loop survival (**G2, the hard gate**) is phase 3.
