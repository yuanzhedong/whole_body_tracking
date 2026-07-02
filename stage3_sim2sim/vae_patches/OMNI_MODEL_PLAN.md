# Omni-modal G1 motion model — architecture & the role of the seed-VAE

*One-pager for the mentor discussion. Goal: an omni-modal model that turns language / audio /
human-reference into Unitree-G1 whole-body motion, physically executed on the robot.*

## The key idea: factor the "brain" into 3 layers; the VAE is the foundation

```
   text / audio / human-ref
            │
   ┌────────▼─────────┐  STAGE 2 — CONDITIONAL GENERATOR     needs PAIRED (modality ↔ motion) data
   │  latent generator│  diffusion (MLD) or AR transformer/MLLM → motion latents / tokens
   └────────┬─────────┘
            │ latent / tokens
   ┌────────▼─────────┐  STAGE 1 — MOTION REPRESENTATION      needs MOTION ONLY (no labels) → the VAE
   │  VAE  (UniMoTok) │  trained on seed (288 h G1); encode↔decode a 128-frame window ↔ latent
   └────────┬─────────┘
            │ decode → G1 reference motion (qpos)
   ┌────────▼─────────┐  STAGE 3 — PHYSICAL EXECUTION
   │   HoloMotion     │  RL tracker → physics (500 Hz) → sim2sim → real G1
   └──────────────────┘
```

The **omni-model = Stage 2 + the VAE's latent space (Stage 1) + HoloMotion (Stage 3).** "seed + VAE" builds
**Stage 1** — and doing it first is the right sequencing.

## Why "seed + VAE first" (the mentor's instinct, justified)
1. **It splits two problems that need different data.**
   - *What G1 motion looks like* → abundant **unlabeled** motion → all of seed (288 h). The **VAE** learns this.
   - *How to map text/audio → motion* → **scarce paired** data → Stage 2, and it's far easier once motion lives
     in a small, smooth latent instead of raw 41-D/125-D motion.
2. **Cheap generation.** Sampling a small latent (then decoding) is much faster than OMG-style motion-space
   diffusion — good for the 30 Hz real-time planner.
3. **Modular & matches UniMoTok.** UniMoTok is a tokenizer feeding MLLMs/VLAs: build the tokenizer on seed,
   then add text/audio/ref conditioning *without retraining the representation*.

## The single make-or-break property
**The VAE's reconstruction quality is a hard ceiling on the entire system.** If the VAE blurs G1 motion, no
Stage-2 generator can ever beat its own decoder. So **milestone #1 is reconstruction**, not text-to-motion:
> Train the VAE on seed → can it faithfully reconstruct held-out G1 motion? (metric: reconstruction MPJPE +
> foot-slide; visually crisp decoded clips.)

If yes → proceed to Stage 2. If the 256-D global latent blurs fast motion → switch to **VQ-VAE/FSQ** (UniMoTok
supports it) or a less-compressed latent.

## Relationship to the OMG model we reproduced
| | OMG (reproduced) | This plan |
|---|---|---|
| Brain | **one-stage**: diffusion directly in 125-D motion space | **two-stage**: VAE latent + a latent generator |
| Pros | simple, high quality | faster sampling, data-efficient conditioning, reusable representation |
| Cons | slow sampling, entangled | extra stage; VAE reconstruction is a ceiling |
They're **compatible**: reuse OMG's text/audio/human-ref conditioning + classifier-free guidance as the Stage-2
generator, operating *in the VAE latent*. Same HoloMotion execution (Stage 3) we already validated (walk/dance/
fight/jump execute upright; the data-scaling experiments showed more seed data → more trackable references).

## Concrete data path (implemented)
seed G1 CSV → `stage2/seed_to_artifacts.py` (cm→m, Euler→quat, 120→30 fps, **joint permutation** to UniMoTok's
order) → `stage2/export_g1_motion.py --to_yup` → 41-D dataset (`6D root-rot + root lin/ang-vel + 29 joints`) →
UniMoTok `config_g1_mldvae` training on the Blackwell GPUs (cu128).

## Roadmap & status
1. **Stage 1 (now):** seed → 41-D → train **G1 MLD-VAE**, measure **reconstruction**. *De-risk in progress;
   small-data run on 2× Blackwell, then scale to full seed (142k clips).*
2. **Stage 2 (next):** conditional latent generator (latent diffusion à la MLD, or tokens + transformer/MLLM).
3. **Stage 3 (have):** decode → HoloMotion → sim2sim → G1. *Validated.*

**Bottom line:** "seed + VAE" is not a detour — it's building the floor the omni-model stands on, from the
biggest, cheapest data available (seed). Gate it on reconstruction quality, then layer conditioning on top.
