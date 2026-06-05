# UniMoTok vs. the BeyondMimic Figure-7 VAE

Analysis of whether **UniMoTok** (Juze Zhang's *Unified Motion Tokenizer*, built on **UniTok**, arXiv:2502.20321) can serve as the **VAE in BeyondMimic Figure 7** (the Stage-2 latent-diffusion controller). Grounded in the BeyondMimic PDF (§S3, Tables S6–S7) and the UniMoTok source.

> **TL;DR:** UniMoTok is an **offline, windowed motion *autoencoder*** (encode a motion clip → latent → *reconstruct the motion*). The Fig-7 VAE is a **per-step conditional *control* VAE** (encode the reference → latent; decode latent **+ proprioception → action**; trained **online via DAgger** in sim at 25 Hz). They do different jobs, so UniMoTok cannot drop in unchanged. It *can* be used, but only via (A) a generate-then-track pipeline, or (B) transplanting its encoder/decoder arch into our online control-VAE trainer.

---

## 1. What UniMoTok is / does

A **unified motion tokenizer** for human / biomechanics / robot motion, built on the UniTok image-tokenizer idea (multi-codebook quantization). It learns a compact latent (continuous **VAE** or discrete **VQ-VAE**) of *motion sequences*, used downstream for motion generation, MLLM/VLA integration, and conversational agents.

- **Repo note (important):** the cloned `main` branch is **incomplete** — the entire `multimodal_tokenizers/models/` folder is *git-ignored* (commit `15efb20` on the dev branch is literally "remove ignoring models folder"). The working code is on branch **`feat/biomechanics_tokenization`** (model wrappers, `build_model`, the MLD VAE, a training guide `docs/mld_vae_training.md`, a motion viewer). **Use that branch.**
- **Two VAE architectures:**
  - **Conv1d VAE** (`archs/lom_vq.py:VAEConv`, `archs/motion_encoder.py`): frame-level latent codes with temporal downsampling (128 frames → ~8 codes).
  - **MLD Transformer VAE** (`archs/mld_vae.py:MldVaeBiomechanics`, from MotionGPT3/MLD): a **SkipTransformer** encoder/decoder that compresses a whole window → a few **global latent tokens** (default `[1, 256]` = 1 token, 256-D for the entire 128-frame clip), then reconstructs. ~17.7 M params. **Self-contained (no pytorch3d/flash-attn for the arch).** This is the modern path and the same **MLD lineage BeyondMimic's VAE is based on**.
- **I/O (both archs):** `encode(motion [B,T,D]) → z`, `decode(z) → motion_hat [B,T,D]`. It **reconstructs the input motion**.
- **Training:** **offline reconstruction** (PyTorch-Lightning + Hydra). Losses are feature-space: feature-MSE + velocity + root-orient + KL (SMPL-X FK only for some *human* metrics; the VAE itself is FK-free, and `LOSS.ABLATION.USE_GENMO_RECONS=false` falls back to plain feature+KL).
- **Representation:** human SMPL-X/GENMO (145-D / 263-D HumanML3D) or biomechanics (49-D), body-part compositional (face/upper/lower/hands). **No robot/G1 representation exists yet.**
- **Data:** dir of clips (`.npz`/`.mot`) + per-dim normalization stats; windowed (`window_size=128`, normalize required). Configs point to cluster paths (`/simurgh2/...`).

## 2. What the BeyondMimic Fig-7 VAE is (recap)

From the PDF (Fig 7B-i, §S3, Table S6):
- **Encoder:** `z = E(ψ, e_anchor)` — input is **only the reference-motion** (phase + anchor error). Latent dim **32**.
- **Decoder = a control policy:** `â = D(z, [g, V_imu, θ, θ̇, a_last])` — combines the latent with **proprioception** to output the **action** (joint targets).
- **Training:** **DAgger** distillation of the tracking policies, modified ELBO `‖â − a_teacher‖² + β·KL` (β=0.01). Runs **per control step, causally, at 25 Hz** as a real-time controller.
- **Why a VAE at all:** to give the **latent diffusion** (Fig 7B-ii) a *smooth* action-latent (the raw action space has "sharp torque spikes" that break diffusion).

## 3. Why UniMoTok is not a drop-in — three axes of mismatch

| axis | UniMoTok VAE | Fig-7 VAE |
|---|---|---|
| **decoder output** | reconstructs **motion** (`z → motion`) | produces an **action** (`z, proprio → action`) |
| **conditioning** | **unconditional** autoencoder (decode from z only) | **conditioned on proprioception** (state-dependent control) |
| **temporal / training** | **offline**, **windowed** (whole 128-frame clip → global tokens), trained by **reconstruction** | **online**, **per-step / causal**, real-time **25 Hz**, trained by **DAgger in the sim env** |

The first row is the deepest: Fig-7's decoder is a *closed-loop controller* that needs the robot's current state to emit the right torque-setpoints; UniMoTok's decoder just reproduces the motion it encoded. The third row is the next hardest: UniMoTok is a Lightning *offline* trainer with no simulator in the loop, whereas Fig-7's VAE is distilled *online* against a teacher policy inside Isaac.

## 4. How it *could* be used (two paths)

- **(A) Generate-then-track (natural fit for UniMoTok).** Use UniMoTok to tokenize G1 **motion**; train the diffusion over its motion-latents; decode latent → motion; let the **Stage-0 tracking policy execute** that motion. Coherent versatile control, reuses our policies, uses UniMoTok as designed — but **not** Fig-7's fused latent→action diffusion (generation and control are separate stages).
- **(B) Arch-transplant into the Fig-7 control VAE (faithful, heavy).** Reuse UniMoTok's encoder/decoder *architecture* but rewire it to the conditional-control form (`decode(z, proprio) → action`), make it per-step/causal, and train it via **DAgger in Isaac** (i.e., inside our `stage2/distill_vae.py`, replacing the MLP backbone). Matches Fig 7, but at that point you're borrowing UniMoTok's *arch*, not running the UniMoTok framework.

**Recommendation:** if the goal is *faithful Fig-7*, the per-step conditional control VAE we already built (`stage2/distill_vae.py`, MLP, DAgger) **is** that VAE; UniMoTok would only contribute a fancier backbone (path B). If the goal is to *leverage UniMoTok as designed*, path A (generate-then-track) is the right architecture — and it's close to how UniMoTok/OmniMM are meant to be used.

## 5. Standalone-run status (for "reproduce correct results")
- **Branch:** must be on `feat/biomechanics_tokenization` (done) — `main` is missing `models/`.
- **VAE that runs:** the **MLD VAE** (`BioMechanicsTokenizer` + `MldVaeBiomechanics`), losses feature-space, arch self-contained.
- **Blocker:** the datasets (`/simurgh2/datasets/AMASS_SMPLH_Bio_20fps`, HumanML3D) are **cluster paths, absent here**. Reproducing *their* numbers needs that data. A **synthetic-data smoke** can still verify the framework + arch + training loop run end-to-end (loss decreases). HumanML3D is publicly downloadable if exact reproduction is wanted later.
