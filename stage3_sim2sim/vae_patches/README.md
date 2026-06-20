# UniMoTok (VAE) patches for the seed→VAE→sim2sim pipeline

These capture the new-pipeline changes that live in the **UniMoTok submodule**
(`Juzezhang/UniMoTok` @ `wbt-integration`). They're vendored here as reviewable files
because the submodule is a third-party repo we don't push to. Apply them in the submodule.

## 1. Loss fix — `biomechanics_tokenizer_rootrot6d_lossfix.patch`
The G1 41-D feature puts the root's heading-canonical **6D rotation at dims [0:6]**, but the
reconstruction loss only supervised `[9:41]` for non-biomech features → root orientation was
**left unsupervised** (decoded root tilt/facing was garbage; normalized MSE 2.47 → worse than
the mean). The patch adds a `motion_dim == 41` branch that supervises the **full** feature
(root_rot6d normalized MSE 2.47 → 0.048). Apply:
```
cd UniMoTok && git apply ../stage3_sim2sim/vae_patches/biomechanics_tokenizer_rootrot6d_lossfix.patch
```

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
