# Spec: G1 robot motion as a new OmniMM modality (Framing 1)

Concrete plan to feed **BeyondMimic stage-1 output (G1 robot motion)** into **OmniMM**
(the unified large motion model in the slides: Yao He, Juze Zhang et al.) as a new
continuous modality, via its own **UniMoTok VAE**.

```
BeyondMimic (track AMASS/LAFAN → G1)  ─►  G1 motion npz  ─►  [export]  ─►  G1 feature clips [T, D_g1]
                                                                                  │
                                                                       G1-VAE (UniMoTok MldVae, nfeats=D_g1)
                                                                                  │ encode → latent tokens
                                                                                  ▼
        OmniMM  ◄── text ──► human-motion ──► biomech ──► [NEW: robot-motion branch]
```

OmniMM's own recipe (from the deck): *"same recipe to add new modalities — the actual problem
becomes **defining representation and getting data**"*, and every continuous modality enters as
*"raw input → VAE → linear projector to text embedding space."* This spec nails down the
representation, the data export, and the VAE; the OmniMM branch wiring is the one piece that
lives in Yao He's repo.

---

## 1. The G1 motion representation

**Grounded facts (this repo):**
- **29 actuated DoF**, fixed order (from `scripts/csv_to_npz.py`): L-leg 6 (hip pitch/roll/yaw, knee, ankle pitch/roll), R-leg 6, waist 3 (yaw/roll/pitch), L-arm 7 (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw), R-arm 7.
- **Floating base** = `pelvis` (body index 0); **tracking anchor** = `torso_link`.
- **Rate:** `sim.dt = 0.005` (200 Hz) ÷ `decimation 4` = **50 Hz** control; motion npz default `output_fps = 50`. OmniMM convention is **20 fps, y-up** (Isaac is z-up).

**Feature vector** (mirrors OmniMM's 52-D biomech *velocity* representation, which deliberately
uses 6-D rotation + root-velocity to avoid the Euler discontinuity the deck hit at 49-D):

| slice | content | dim | note |
|---|---|---|---|
| `0:6`  | root (pelvis) orientation, 6-D (Zhou et al. 2019) | 6 | continuous, no gimbal discontinuity |
| `6:9`  | root linear velocity, root-local frame | 3 | velocity rep (integrate to recover translation) |
| `9:12` | root angular velocity, root-local frame | 3 | |
| `12:41`| 29 joint angles, fixed order above | 29 | native G1 actuation |

→ **`D_g1 = 41`** (minimal). Variants: drop root ang-vel → 38; extend with 29 joint
velocities → 70 (deck's biomech VAE "extends with velocity, acceleration").

**Why this is clean for G1:** unlike the BSM biomech rep (which needs SMPL→marker→OpenSim
fitting), G1 joint angles *are* the robot's actuated DoF — no FK fitting, no skeleton ambiguity.
Body-link positions are redundant (recoverable by FK through the G1 URDF), so we keep the rep
compact and use FK only for an eval metric (joint-distance), as the deck does.

## 2. Data export: BeyondMimic npz → G1 feature clips

**Real npz schema** (`scripts/csv_to_npz.py`, the canonical motion file): `fps`,
`joint_pos [T,29]`, `joint_vel [T,29]`, `body_pos_w [T,B,3]`, `body_quat_w [T,B,4]` (wxyz),
`body_lin_vel_w [T,B,3]`, `body_ang_vel_w [T,B,3]`. Body 0 = pelvis.

**Exporter** (`stage2/export_g1_motion.py`, buildable now):
1. Load npz; take pelvis = body 0: `body_quat_w[:,0]`, `body_lin_vel_w[:,0]`, `body_ang_vel_w[:,0]`.
2. `quat(wxyz) → 6-D rotation`; rotate lin/ang velocity into the yaw-aligned root-local frame.
3. `joint_pos[:, :29]` in the fixed order → angles slice.
4. Concatenate → `[T, 41]`.
5. **z-up → y-up** transform (Isaac → OmniMM; the deck explicitly fixed a y-up bug here — get the basis change right).
6. **Resample 50 → 20 fps** to match OmniMM's VAE convention.
7. Window into **128-frame** clips; compute per-dim **mean/std** over the corpus; save a dir of clips + stats (the format `BioMechanicsDataModule` expects).

**Data source:** BeyondMimic already tracks retargeted AMASS→G1 + LAFAN1 (our earlier verified
suite). Each tracked motion's npz → one G1 clip set → thousands of windows. **Mirror augmentation**
is free (G1 is bilaterally symmetric: swap L/R joint groups + flip roll/yaw signs) — doubles data,
as the deck does for biomech.

## 3. The G1-VAE (UniMoTok)

Clone `configs/config_biomechanics_mldvae_velocity_nomirror_v2.yaml` → `config_g1_mldvae.yaml`:
- arch `MldVaeBiomechanics`, **`vae_test_dim: 41`**, `latent_dim: [1, 256]`, `num_layers: 9`, `ff_size: 1536`, `num_heads: 8` (unchanged — arch is generic over `nfeats`; verified instantiates+trains via the smoke).
- **datamodule:** new `G1DataModule` mirroring `BioMechanicsDataModule`, pointed at the exported clip dir + stats; `window_size 128`, `normalize_motion true`.
- **losses:** feature smooth-L1 + velocity + KL (1e-5). Replace the BSM Euler `root_orient` geodesic with a **6-D rotation loss** (we use 6-D root rot, not Euler). Keep `root_velocity` loss on the `6:9` slice.
- **known limit (deck):** VAE is valid to ~6 s / 128 frames; longer is OOD. At 20 fps, 128 frames = 6.4 s.

Train: `python -m training.train_tokenizer --cfg configs/config_g1_mldvae.yaml --nodebug`.
Success = reconstruction loss drops; joint-angle recon error small (deck's biomech target was
joint angles < 0.1 rad). All on the `wbt-integration` branch of `Juzezhang/UniMoTok`.

## 4. OmniMM branch integration (lives in Yao He's repo — out of our control)

The deck's recipe, applied to robot-motion:
- Add a **MoT branch** for "robot-motion": `G1-VAE encode → linear projector → text-embedding space`; decode = `hidden states → latent → G1-VAE decode → G1 motion`.
- Register a dataset binder entry (`TextRobotMotionDataset` / `MotionRobotDataset`) alongside the text/motion/biomech ones.
- Set the projector dims (`256 → model dim`) and the attention masks (motion-style token fusing; causal/non-causal per the deck's "Mixed Self-Attention").
- **New tasks unlocked:** T2R (text→robot-motion), R2T, M2R/R2M (human↔robot motion), and the cross-grounding the deck motivates ("different modalities mutually grounded").

What OmniMM needs from us: the **trained G1-VAE checkpoint** + the **representation/normalization
stats** + the **latent-standardization stats** + this spec. We hand those over; they wire the branch.

**Generative-readiness gate (`sim2sim_vae_eval.py` Phase 3, no Isaac).** Because the diffusion stage
GENERATES latents and decodes them (it never feeds the VAE a real encoding), reconstruction quality
is necessary but not sufficient. Phase 3 checks the three properties diffusion actually needs:
(A) aggregated posterior ≈ N(0,I); (B) prior-sampled z~N(0,I) decode to in-distribution, joint-valid,
smooth motion; (C) latent interpolation stays smooth/in-distribution. Result on EX_T4w_base (9-clip,
latent-128): **B + C PASS** (decoder is healthy off-manifold — smooth, valid, interpolable) but
**A FAILS** — aggregate is mean 0.65 / std 1.68, not N(0,I). KL 5e-6 vs 5e-5 are identical here (both
in the negligible-KL regime), so this is NOT fixed by the existing KL sweep. **Fix = ship the per-dim
latent-standardization vectors** (Phase 3 emits `latent_standardization.{mean,std}`, length=latent_dim);
latent diffusion rescales latents to ~unit variance anyway (cf. Stable Diffusion's 0.18215 factor),
so a non-unit aggregate is a "hand over the stats" item, not a retrain blocker. Only retrain with
orders-of-magnitude higher KL if a *natively* N(0,I) latent is required.

## 5. Decisions & risks

- **Root body:** pelvis (free base, recommended) vs torso_link (tracking anchor). Use pelvis.
- **fps:** resample to 20 (drop-in with OmniMM) vs keep 50 (richer, but model must be rate-consistent). Recommend 20.
- **z-up→y-up:** mandatory basis change; the deck shows a bug here cost them — verify with a render before training.
- **Root drift / foot-skate:** velocity-rep root integrates with drift absent contact constraints (deck notes this); acceptable for v1, add contact later.
- **Window length:** 128 frames; pick fps so the window matches OmniMM's other modalities.
- **Redundancy:** joint angles only (compact); body positions via FK for an eval metric (joint-distance to GT), not in the input.

## 6. Build order (what I can do now vs. blocked)

| step | artifact | status |
|---|---|---|
| 1 | `stage2/export_g1_motion.py` — WBT npz dir → G1 `[T,41]` clips + mean/std | **buildable now** |
| 2 | UniMoTok `config_g1_mldvae.yaml` + `G1DataModule` (mirror biomech) | **buildable now** |
| 3 | train G1-VAE on real BeyondMimic clips → verify recon | after 1–2 + a teacher motion corpus |
| 4 | OmniMM robot-motion branch wiring | **blocked** — needs Yao He's OmniMM repo |

**Recommended first step:** build (1) the exporter and run it on the existing tracked-motion
suite, so we have a real G1 clip dataset; then (2) the G1-VAE config + a training smoke on it.
Those two prove the representation + data path end-to-end before anyone touches OmniMM.
