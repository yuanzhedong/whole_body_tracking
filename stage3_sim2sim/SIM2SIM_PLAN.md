# Stage 3 — sim2sim pipeline: build & verify

```
BONES-SEED G1 motion → UniMoTok VAE → decode → HoloMotion tracker → MuJoCo physics
(142k clips, 288h,      encode→latent   qpos_36   RL policy tracks    does G1 stay
 motion-only)                                     the decoded ref     upright & on-trace?
```

**Only one new component:** `decode_to_qpos36` (41-D feature → `qpos_36`). Everything from
`qpos_36` onward (`export_holomotion_deployment_clip`, the HoloMotion ONNX tracker, MuJoCo, and the
`tracker_executed` metrics) already exists and was validated in the OMG reproduction.

## Components & contracts
| | component | contract | status |
|---|---|---|---|
| C1 | `stage2/export_g1_motion.build_features` | world motion → 41-D | exists |
| **C2** | **`stage3_sim2sim/decode_to_qpos36`** | **41-D → qpos_36** | NEW |
| C3 | UniMoTok VAE | 41-D → 41-D | trained |
| C4 | `export_holomotion_deployment_clip` | qpos_36 → deploy npz (30→50 Hz) | exists |
| C5 | HoloMotion ONNX tracker | deploy npz → MuJoCo actions | exists |
| C6 | `run_tracker_executed_benchmark` | rollout → survival, g_mpjpe, mpjpe, e_vel/e_acc | exists |

## Test pyramid & gates
- **L0 unit** — rotation 6D↔matrix, yaw/position integration, `decode_to_qpos36` layout, and the
  **keystone synthetic round-trip** (`qpos→build_features→invert→recover`). **Gate A.**
- **L1 pairwise** — real-clip round-trip (joint MAE < 0.5°, report root drift); VAE↔inverse; inverse↔HoloMotion export schema.
- **L2 multi** — **Gate B** validate-the-validator (original motion survives in MuJoCo); **Gate C** decoded motion survives.
- **L3 e2e** — full chain on N held-out clips; **Gate D** decoded-vs-original survival/mpjpe gap.

Gates are stop-and-verify: do not proceed past a red gate.

## Status
- **Gate A — PASSED.** L0 (8 tests): joints recovered exactly; orientation ≤ 0.32°; root drift ≤ 1.1 mm
  over a 128-frame window (synthetic smooth motion). Inverse math is faithful.
- L1 / L2 / L3 — pending.

## Main risk
Root integration drift (C2 rebuilds world trajectory from velocities). L0/L1 measure it; if real-clip drift
is large we additionally report **joint-only** and **hybrid (decoded joints + original root)** sim2sim so a
drift artifact never masquerades as a VAE failure.

## The inverse (C2) derivation
The forward 41-D map stores `[root_rot6d(6), root_lin_vel_local(3), root_ang_vel_local(3), joints(29)]`,
heading-canonicalized (yaw stripped) in a y-up frame. The inverse:
1. **joints** = features[12:41] (exact passthrough).
2. **R_canon** = Gram-Schmidt of the two stored columns (root_rot6d) → tilt/pitch/roll, exact.
3. **yaw(t)** = integrate the up-axis component of root_ang_vel_local (exact up-component because the
   heading rotation preserves that axis), seeded at yaw0.
4. **R(t)** = R_strip(yaw) · R_canon; **root world velocity** = R_strip · root_lin_vel_local.
5. **root_pos(t)** = trapezoidal integration of world velocity, seeded at pos0.
6. Undo the y-up basis change(s) → z-up `qpos_36 = [root_pos, root_quat_wxyz, joints]`.
Only heading and world translation are integrated (→ bounded drift); pitch/roll come exactly from
root_rot6d, so balance-relevant orientation is exact.
