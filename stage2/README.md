# `stage2/` — BeyondMimic Stage-2 reimplementation (WIP)

Reimplementing BeyondMimic's **versatile-control** stage (paper Fig 7B; *not* in the open repos)
on top of the Stage-0 tracking policies this repo trains. Latent **state-action diffusion**:

```
RL tracking policies (Stage 0, this repo)
   │  Phase 1: DAgger-distill into a conditional VAE  ← IMPLEMENTED HERE
   ▼
smooth latent (dim 32) + decoder policy
   │  Phase 2: collect VAE rollouts (OU-noise error band)
   ▼
state-latent trajectories
   │  Phase 3: train latent diffusion (transformer denoiser)
   ▼
   │  Phase 4: classifier guidance (cost-gradient steering) at test time
   ▼
versatile control (joystick / waypoint / obstacle / inpainting)
```

## Phase 1 (here): VAE distillation
- `vae_model.py` — conditional `MotionVAE`: `encode(reference_obs)→z(32)`, `decode(z, proprio_obs)→action`.
  Table S6: latent 32, enc/dec MLP `[2048,1024,512]` ELU, KL β=0.01.
- `distill_vae.py` — DAgger loop in `Tracking-Flat-G1-v0`: roll out a teacher→student mixture,
  supervise the VAE with the teacher action via the modified ELBO
  `‖â−a‖² + β·KL(q(z)‖N(0,I))`. Splits the policy obs into reference terms (command/phase +
  anchor error → encoder) and proprioceptive terms (base/joint vel, joint pos, last action → decoder).

Run (on a 4090):
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
  .venv/bin/python stage2/distill_vae.py \
    --teacher_ckpt logs/rsl_rl/g1_flat/<walk_run>/model_29999.pt \
    --motion_file /tmp/wbt_fix/walk.npz --iters 10000 --out stage2/out/vae_walk.pt
```

## Fidelity notes / deviations from the paper (so far)
- **Control rate:** paper runs Stage-2 at **25 Hz** (`decimation=8`); our teachers are 50 Hz. The
  scaffold uses the teacher as-is (50 Hz) to prove the machinery — switch the teacher to 25 Hz for
  full fidelity.
- **Single teacher (walk):** paper distills *many* policies into one VAE; we start with one (walk).
- **Not yet:** sagittal-symmetry augmentation, OU-noise data collection (that's Phase 2), the
  emphasis projection / character-frame state featurization (Phase 2/3).
- `out/` is git-ignored.

## Phase-1 verification gate (`verify_vae.py`) — run BEFORE the diffusion
A distilled VAE can pass action-MSE yet still drift/fall closed-loop, or have a degenerate latent
that's useless for diffusion. So we gate on 4 checks, each vs the teacher:

| gate | proves | metric | pass |
|---|---|---|---|
| **G1** reconstruction | decoder learned the teacher map | `‖decode(μ,proprio) − a_teacher‖` | small MSE |
| **G2** closed-loop *(hard gate)* | VAE *policy* controls the robot | success rate, E_mpbpe, E_mpjpe vs teacher | success ≥ 0.9× teacher, E_mpbpe ≤ 1.3× |
| **G3** latent structure | latent smooth, **used**, ~N(0,I) | agg-posterior, active dims, **z-ablation** `‖decode(E(ref))−decode(0)‖` | z-ablation ≥ 0.1, ≥2 active dims |
| **G4** robustness | recovery basin for Phase-2 OU noise | closed-loop + action noise → fall rate | recovers |

**Do not proceed to the diffusion until G2 + G3 pass.** G3's z-ablation is critical: if the decoder
ignores the latent, the diffusion (which models that latent) has nothing meaningful to generate.
```bash
.venv/bin/python stage2/verify_vae.py --vae stage2/out/vae_walk.pt \
    --teacher_ckpt <walk model_29999.pt> --motion_file /tmp/wbt_fix/walk.npz
```

## Verification ladder for the whole Fig-7 reproduction (each sub-step gated)
- ✅ Stage-0 tracking: success rate / convergence (already verified, see scorecard report).
- ⏳ **VAE: G1–G4 above** (this phase).
- ☐ Phase-2 data: trajectory coverage + error-band stats.
- ☐ Latent diffusion: *unconditional* generation → decode → rollout (does generated motion track?).
- ☐ Guidance: per-task success (joystick follows cmd, waypoint reached, obstacle avoided).

## Status
Phase 1 implemented + verification harness. Phases 2–4 are the diffusion side (paper §S3 / Tables
S6–S7: horizon 16, history 4, transformer 512-dim/8-heads/6-layers, 20 denoise steps).
