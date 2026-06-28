# Documentation

Setup and reference docs for the G1 motion-tokenizer pipeline built on top of this BeyondMimic
tracking repo.

## Start here

- **[PIPELINE_SETUP.md](PIPELINE_SETUP.md)** — end-to-end setup & run guide for the
  **seed data → UniMoTok VAE → BFM-Zero → sim2sim** pipeline: environment, exact commands for each
  stage, external prerequisites, verification gates, and the W&B result reports.

## Pipeline at a glance

```
BONES-SEED G1 motion ─► 41-D features ─► UniMoTok MLD-VAE ─► decode ─► qpos_36 ─► BFM-Zero / HoloMotion ─► MuJoCo
   stage2/seed_to_       stage2/export_     UniMoTok          stage3_sim2sim/      stage3_sim2sim/         survival /
   artifacts.py          g1_motion.py       (submodule)       decode_to_qpos36.py  run_l3_eval.py          tracking error
```

Stages 1–3 are self-contained in this repo; Stage 4 (the physics validator) shells out to the OMG
and BFM-Zero repos — see [PIPELINE_SETUP.md §0](PIPELINE_SETUP.md#0-what-is-and-isnt-self-contained).

## W&B reports (results)

- [Headline pipeline report — BONES-SEED → UniMoTok VAE → BFM-Zero sim2sim](https://wandb.ai/toddler_tracking/g1-sim2sim/reports/x--VmlldzoxNzMzOTg2Mg==)
- [HoloMotion-validated pipeline (sibling)](https://wandb.ai/toddler_tracking/g1-sim2sim/reports/x--VmlldzoxNzMwNzgxMw==)
- [Tracker comparison — HoloMotion vs BFM-Zero](https://wandb.ai/toddler_tracking/g1-sim2sim/reports/x--VmlldzoxNzMzNDI0MA==)

## Related design docs (in-tree)

- `stage3_sim2sim/SIM2SIM_PLAN.md` — sim2sim gates, the 41-D inverse (C2) derivation, hybrid-root rationale.
- `stage2/g1_omnimm_modality_spec.md` — 41-D representation rationale and OmniMM hand-off.
- `stage2/README.md` — Stage-2 VAE-distillation context (BeyondMimic Fig-7 reproduction).
- `../PROGRESS.md` — reverse-chronological progress log.
