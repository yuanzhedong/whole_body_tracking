# Pipeline correctness verification — per-component I/O contracts, corner cases, e2e

Goal: prove **every submodule** is correct via explicit input → expected-output checks (known-answer
where possible), cover corner cases, then verify the whole chain end-to-end. Legend: ✅ covered, 🔨 to add.

```
CSV ─C1─► artifact ─C2─► 41-D feature ─C3(VAE)─► 41-D rec ─C4─► qpos_36 ─C5─► deploy npz ─C6─► tracker ─C7─► MuJoCo ─C8─► metrics
```

## C1. Data ingest — `seed_to_artifacts` / `export_g1_motion.build_features`
**Contract:** CSV[N,36] @120fps → artifact {joint_pos[T,29], body_pos_w[T,30,3], body_quat_w[T,30,4], fps=30}.
- Known-answer: constant-pose CSV → constant joints, identity quat; one row of known Euler → known quat. 🔨
- cm→m: root translate /100 exact. 🔨  | deg→rad joints exact. 🔨
- 120→30 resample = stride-4 (len math). 🔨  | OMG↔UniMoTok joint permutation is a **bijection** (perm is a permutation of 0..28). 🔨
- Corner: clip shorter than min → skipped (returns None); NaN row tolerated/skipped; mirrored `_M` variant. 🔨

## C2. Forward feature map — `build_features` (world → 41-D)
**Contract:** root_pos[T,3], quat_xyzw[T,4], joints[T,29], dt → feats[T,41].
- Static clip (no motion) → lin_vel≈0, ang_vel≈0, rot6d constant, joints constant. 🔨
- Pure +yaw rotation → rot6d constant (heading stripped), ang_vel only on up-axis, lin_vel≈0. 🔨
- Pure translation → joints/rot6d constant, lin_vel = velocity, ang_vel≈0. 🔨
- Quat double-cover: q and −q give identical features. 🔨
- Non-normalized quat input → tolerated (scipy normalizes). 🔨

## C3. Inverse — `decode_to_qpos36` / `invert_build_features`
**Contract:** feats[T,41], dt → (root_pos, quat_xyzw, joints) recovering build_features' input.
- Synthetic round-trip: joints exact, orient ≤0.32°, root ≤1.1mm. ✅
- Real-clip round-trip: joints exact, tilt 2e-7, lin_vel 3.6e-9. ✅
- Off-manifold 6D (noisy) → Gram-Schmidt yields proper rotation (RRᵀ=I, det=1). ✅
- Exact `np.gradient` inverse (integration is exact). ✅
- Corner: zero-velocity feats → constant root; single short window (T<8); yaw wrap ±π continuity. 🔨
- KNOWN LIMITATION: absolute world-root height (double-yup) → strict xfail. ✅

## C4. VAE encode/decode — `vae_decode_clip.decode_features`
**Contract:** feats[T,41] + (mean,std) → rec[T,41], same shape, finite.
- Determinism: eval mode, same input → same output (twice). 🔨
- Shape/finite on window-length, shorter-than-window (pad), longer (chunked). 🔨
- Normalization handled (no NaN when std has zeros → clipped). 🔨
- Joints block is the bulk of the signal (rec close to input on a trained clip; informational). ✅(recon_check)

## C5. decode→reference — `features_to_qpos36` / `build_hybrid_qpos36`
- qpos joints (7:36) == feature joints (12:41) exactly; quat normalized. ✅
- Hybrid: root(0:7) from original, joints(7:36) from decoded; length = min. ✅
- Corner: decoded shorter than original → truncate to min; identity case. 🔨

## C6. HoloMotion export — `export_holomotion_deployment_clip` (OMG, trusted dep)
**Contract:** qpos_36 + fps → deploy npz {ref_dof_pos[T',29], ref_dof_vel, ref_global_translation/rotation, fps=50}.
- Schema/keys/shapes present; ref_dof_pos == qpos[:,7:36] (post-resample); 30→50 frame-count math. 🔨(integration)

## C7. Tracker obs — `build_holomotion_obs` + ONNX (OMG, trusted dep)
**Contract:** obs = 522-d (132 proprio + 10×39 future ref); action[29] → target = default_pose + scale·action.
- obs dimension == 522 on a real rollout; actions finite. 🔨(integration assert)

## C8. MuJoCo physics (OMG, trusted dep)
- Validate-the-validator: ORIGINAL motion survives (root z > 0.4) — Gate B. ✅
- Determinism: same seed/input → same executed qpos. 🔨(integration)

## C9. Metrics — `rollout_metrics`
- identical exec/ref → survival 1, joint_rmse 0, drift 0. ✅
- known fall (z<0.4) → survival 0; partial fall → fraction. ✅
- known joint offset / root offset → exact value. ✅
- Corner: length mismatch (exec≠ref) → uses min; single frame. 🔨

## E2E
- Full chain (CSV/artifact → … → metrics) on a known clip → survival>0.9 (Gate B). ✅
- Decoded-vs-original across 6 motion types (Gate D). ✅
- **Reproducibility:** same clip+ckpt → same decoded qpos + same survival. 🔨
- **Failure injection (negative test):** corrupt decoded joints (+0.5 rad noise) → survival drops / tracking error rises — proves the metric actually detects bad motion. 🔨

## Execution order
1. C1 ingest known-answer + permutation bijection.
2. C2 forward known-answer (static / pure-yaw / pure-translation / quat double-cover).
3. C3/C4/C5 corner cases (zero-vel, short window, determinism, chunking, length mismatch).
4. C9 metric corner cases.
5. C6/C7/C8 contract/integration asserts (skip if assets absent).
6. E2E reproducibility + failure injection.
