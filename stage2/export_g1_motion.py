"""Step 1 of the G1->OmniMM modality pipeline (see stage2/g1_omnimm_modality_spec.md).

Exports BeyondMimic motion clips (artifacts/<name>:v0/motion.npz) into a UniMoTok-ready
G1 motion dataset:
  - 41-D heading-canonicalized *velocity* representation per frame (matches OmniMM's
    biomech rep style: 6-D root rot + root lin/ang vel + 29 joint angles),
  - resampled 50 -> target fps,
  - split by clip (or by category) into train/val/test (no window leakage),
  - optional quality filter on per-clip tracking metrics,
  - **over-preserved FK ground truth** (body_pos_w, joint_pos, root_quat) so the eval
    (FK-MPJPE, joint-angle error, joint-limit %) never needs a re-export,
  - per-dim mean/std computed on TRAIN frames only.

NOTE ON DATA SOURCE: artifacts/*/motion.npz are the *reference* (retargeted) motions the
tracking policy imitates -- not the policy's executed rollout. For v1 we train on the
reference and rely on the quality filter (only keep clips the policy tracks well, so the
reference is provably realizable). A later flag can point at policy-rollout npz instead.

Run (no Isaac needed -- pure numpy/scipy):
  .venv/bin/python stage2/export_g1_motion.py --artifacts_dir artifacts \
      --out_dir stage2/out/g1_dataset --target_fps 20
  # with a quality filter produced by a Stage-0 eval pass:
  #   ... --quality_json stage2/out/track_quality.json --min_survival 0.95 --max_mpbpe_mm 50
"""
import argparse, os, json, glob, hashlib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Canonical native order (matches motion.npz joint_pos / body_pos_w; captured from the G1 asset).
JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
ROOT_BODY_INDEX = 0  # 'pelvis'
FEATURE_LAYOUT = {"root_rot6d": [0, 6], "root_lin_vel": [6, 9], "root_ang_vel": [9, 12], "joint_pos": [12, 41]}
FEATURE_DIM = 41

CATEGORY_KEYWORDS = [  # ordered; first match wins
    ("sprint", "sprint"), ("run", "run"), ("walk", "walk"), ("dance", "dance"),
    ("jump", "jump"), ("fight", "fight"), ("fall", "fall"), ("sport", "sport"),
]


def categorize(name):
    low = name.lower()
    for kw, cat in CATEGORY_KEYWORDS:
        if kw in low:
            return cat
    return "other"


def mirror_pairs():
    """Index pairs (l, r) for left/right joints; used later for mirror augmentation."""
    idx = {n: i for i, n in enumerate(JOINT_NAMES)}
    pairs = []
    for n, i in idx.items():
        if n.startswith("left_"):
            r = "right_" + n[len("left_"):]
            if r in idx:
                pairs.append([i, idx[r]])
    return pairs


def resample_linear(x, t_in, t_out):
    """Linear-interp x[T,...] along axis 0 to t_out."""
    flat = x.reshape(x.shape[0], -1)
    out = np.stack([np.interp(t_out, t_in, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    return out.reshape(t_out.shape[0], *x.shape[1:])


def quat_wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def yaw_of(rotmats):
    """Heading angle about world +z (Isaac is z-up)."""
    return np.arctan2(rotmats[:, 1, 0], rotmats[:, 0, 0])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((a.shape[0], 3, 3))
    R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1.0
    return R


def build_features(root_pos, root_quat_xyzw, joint_pos, dt):
    """root_pos[T,3], root_quat_xyzw[T,4], joint_pos[T,29] -> features[T,41] (heading-canonical velocity rep)."""
    R = Rotation.from_quat(root_quat_xyzw).as_matrix()         # [T,3,3]
    yaw = yaw_of(R)
    Rz_inv = rotz(-yaw)                                          # heading-removal
    R_canon = np.einsum("tij,tjk->tik", Rz_inv, R)              # yaw stripped -> roll/pitch
    rot6d = R_canon[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)  # first two cols flattened

    lin_vel_w = np.gradient(root_pos, dt, axis=0)               # world lin vel
    lin_vel_local = np.einsum("tij,tj->ti", Rz_inv, lin_vel_w)  # heading frame
    rel = Rotation.from_matrix(R[1:] @ np.transpose(R[:-1], (0, 2, 1)))
    ang_w = rel.as_rotvec() / dt
    ang_w = np.concatenate([ang_w[:1], ang_w], axis=0)          # pad to T
    ang_vel_local = np.einsum("tij,tj->ti", Rz_inv, ang_w)

    return np.concatenate([rot6d, lin_vel_local, ang_vel_local, joint_pos], axis=1).astype(np.float32)


def process_clip(path, target_fps):
    d = np.load(path, allow_pickle=True)
    in_fps = int(np.asarray(d["fps"]).reshape(-1)[0])
    T = d["joint_pos"].shape[0]
    dur = (T - 1) / in_fps
    t_in = np.arange(T) / in_fps
    t_out = np.arange(0.0, dur + 1e-9, 1.0 / target_fps)

    joint_pos = resample_linear(d["joint_pos"][:, :29], t_in, t_out)
    body_pos_w = resample_linear(d["body_pos_w"], t_in, t_out)             # FK ground truth (all 30 bodies)
    root_pos = body_pos_w[:, ROOT_BODY_INDEX, :]
    root_q_xyzw = quat_wxyz_to_xyzw(d["body_quat_w"][:, ROOT_BODY_INDEX, :])
    slerp = Slerp(t_in, Rotation.from_quat(root_q_xyzw))
    root_quat_rs = slerp(np.clip(t_out, t_in[0], t_in[-1])).as_quat()      # xyzw, resampled

    feats = build_features(root_pos, root_quat_rs, joint_pos, 1.0 / target_fps)
    raw = {"joint_pos": joint_pos.astype(np.float32), "body_pos_w": body_pos_w.astype(np.float32),
           "root_quat_xyzw": root_quat_rs.astype(np.float32), "fps": np.array([target_fps])}
    return feats, raw, in_fps, T


def assign_split(name, val_ratio, test_ratio):
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 / 1000.0
    if h < test_ratio:
        return "test"
    if h < test_ratio + val_ratio:
        return "val"
    return "train"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts_dir", default="artifacts")
    p.add_argument("--out_dir", default="stage2/out/g1_dataset")
    p.add_argument("--target_fps", type=int, default=20)
    p.add_argument("--window", type=int, default=128, help="window size for the #windows stat / datamodule")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--split_mode", choices=["clip", "category"], default="clip")
    p.add_argument("--holdout_categories", nargs="*", default=[], help="category-mode: these go entirely to test")
    p.add_argument("--quality_json", default=None, help="{clip: {survival, e_mpbpe_mm}}; clips failing thresholds are dropped")
    p.add_argument("--min_survival", type=float, default=0.95)
    p.add_argument("--max_mpbpe_mm", type=float, default=50.0)
    p.add_argument("--to_yup", action="store_true", help="convert z-up->y-up (OmniMM); OFF by default -- VALIDATE with a render first")
    args = p.parse_args()

    if args.to_yup:
        raise NotImplementedError("z-up->y-up not enabled in v1: needs a visual check first (see spec risk note). "
                                  "Dataset is exported in Isaac z-up; convert + render-verify before OmniMM training.")

    quality = json.load(open(args.quality_json)) if args.quality_json else None
    if quality is None:
        print("WARNING: no --quality_json -> tracking-quality filter DISABLED (all clips included). "
              "Generate per-clip Stage-0 eval metrics and re-run with --quality_json before scaling experiments.")

    paths = sorted(glob.glob(os.path.join(args.artifacts_dir, "*", "motion.npz")))
    if not paths:
        raise SystemExit(f"no motion.npz under {args.artifacts_dir}/*/")
    for s in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.out_dir, s), exist_ok=True)

    manifest, train_frames, n_inc, n_drop = {}, [], 0, 0
    for path in paths:
        name = os.path.basename(os.path.dirname(path)).replace(":v0", "")
        cat = categorize(name)
        q = quality.get(name) if quality else None
        if quality is not None:
            if q is None:
                print(f"  drop {name}: no quality entry"); n_drop += 1; continue
            if q.get("survival", 0) < args.min_survival or q.get("e_mpbpe_mm", 1e9) > args.max_mpbpe_mm:
                print(f"  drop {name}: survival={q.get('survival')} mpbpe={q.get('e_mpbpe_mm')}mm"); n_drop += 1; continue

        feats, raw, in_fps, T_in = process_clip(path, args.target_fps)
        if args.split_mode == "category":
            split = "test" if cat in args.holdout_categories else assign_split(name, args.val_ratio, 0.0)
        else:
            split = assign_split(name, args.val_ratio, args.test_ratio)

        np.savez(os.path.join(args.out_dir, split, name + ".npz"),
                 motion=feats, category=cat, **raw)
        if split == "train":
            train_frames.append(feats)
        n_win = max(0, feats.shape[0] - args.window + 1)
        manifest[name] = {"split": split, "category": cat, "n_frames_in": int(T_in),
                          "n_frames_out": int(feats.shape[0]), "in_fps": in_fps,
                          "out_fps": args.target_fps, "n_windows": int(n_win), "quality": q}
        n_inc += 1
        print(f"  {name:38s} {cat:7s} {split:5s} {T_in:6d}@{in_fps} -> {feats.shape[0]:5d}@{args.target_fps}  win={n_win}")

    if not train_frames:
        raise SystemExit("no TRAIN clips -> cannot compute normalization (loosen filter / ratios)")
    allf = np.concatenate(train_frames, axis=0)
    mean = allf.mean(0).astype(np.float32)
    std = np.maximum(allf.std(0), 1e-6).astype(np.float32)
    np.savez(os.path.join(args.out_dir, "normalization.npz"), mean=mean, std=std)

    meta = {"feature_dim": FEATURE_DIM, "feature_layout": FEATURE_LAYOUT, "target_fps": args.target_fps,
            "window": args.window, "up_axis": "z", "root_body_index": ROOT_BODY_INDEX,
            "joint_names": JOINT_NAMES, "mirror_pairs": mirror_pairs(),
            "data_source": "beyondmimic_reference_motion", "quality_filtered": quality is not None,
            "n_included": n_inc, "n_dropped": n_drop, "clips": manifest}
    json.dump(meta, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=2)

    counts = {s: sum(1 for c in manifest.values() if c["split"] == s) for s in ["train", "val", "test"]}
    tot_win = {s: sum(c["n_windows"] for c in manifest.values() if c["split"] == s) for s in ["train", "val", "test"]}
    print(f"\nincluded {n_inc} clips (dropped {n_drop}) -> {args.out_dir}")
    print(f"  split clips:   {counts}")
    print(f"  split windows: {tot_win}")
    print(f"  feature_dim {FEATURE_DIM}, fps {args.target_fps}, up_axis z (y-up conversion pending)")
    print(f"  normalization: mean/std over {allf.shape[0]} train frames")


if __name__ == "__main__":
    main()
