# UniMoTok (VAE) patches for the seed→VAE→sim2sim pipeline

These capture the new-pipeline changes that live in the **UniMoTok submodule**
(`Juzezhang/UniMoTok` @ `wbt-integration`). They're vendored here as reviewable files
because the submodule is a third-party repo we don't push to. Apply them in the submodule.

## 1. Loss fix — `biomechanics_tokenizer_rootrot6d_lossfix.patch`  ⚠️ PROVEN REDUNDANT — NOT applied

**Do not apply this.** It is kept only as reference for the investigation.

We initially thought the G1 root orientation (rot6d, dims [0:6]) was *unsupervised* (the main
reconstruction slice for 41-D is `[9:41]`). That was wrong: **stock UniMoTok already supervises
root orientation** via a **geodesic loss** on dims [0:6] (`root_orient_loss`, gated by
`LAMBDA_ROOT_ORIENT`, which our configs set to 5.0). The misleading signal was that we measured
*raw 6D MSE* (2.47) — but 6D rotation isn't unique, so raw-MSE can be high while the rotation is
correct. The patch (smooth_l1 on raw [0:6]) duplicates supervision the geodesic term already provides.

Proof (gradient flow on the real loss): `stage3_sim2sim/tests/test_unimotok_root_orient.py` —
`|grad on [0:6]|` is 11.10 with `LAMBDA_ROOT_ORIENT=5` and 0.00 with `=0`. So we train on **stock
UniMoTok**; this patch is intentionally *not* applied.

## 2. Training configs — `configs/config_g1_seed_*.yaml`
G1 MLD-VAE configs on the BONES-SEED 41-D features (drop into `UniMoTok/configs/`):
- `config_g1_seed_p1.yaml` — small de-risk run (latent 128, 5 layers).
- `config_g1_seed_full.yaml` — full 142k clips (latent 256, 9 layers).
- `config_g1_seed_512.yaml` / `config_g1_seed_1024.yaml` — capacity sweep.
- `config_g1_seed_512_fixed.yaml` — **lat-512 on the root-fixed features** (the recommended
  Stage-1 config; `data_dir = g1_seed_full_yup_fixed`).

Run: `CUDA_DEVICE_ORDER=PCI_BUS_ID .venv_umt/bin/python -m training.train_tokenizer --cfg configs/<cfg> --device 0 3 --nodebug`

## 3. `OMNI_MODEL_PLAN.md`
The 3-stage omni-model architecture write-up (VAE = Stage 1; this pipeline builds + validates it).

## Capacity-sweep finding (for context)
latent 256 / 512 / 1024 all converge to the same ~13° dynamic-reconstruction floor → the
dynamic ceiling is **architectural** (global single-token over-compression), not capacity.
**lat-512** is the best speed/quality config and gives the best sim2sim survival.
