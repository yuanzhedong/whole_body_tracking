"""Convert BONES-SEED G1 CSVs -> artifacts/<name>/motion.npz for export_g1_motion.py.

Seed CSV (36 cols, 120fps): Frame, root_translate{XYZ}(cm), root_rotate{XYZ}(Euler deg),
29 joints(deg) in OMG/G1_JOINT_NAMES (sequential) order.

export_g1_motion.process_clip wants: joint_pos[T,>=29], body_pos_w[T,30,3] (z-up, idx0=pelvis),
body_quat_w[T,30,4] (wxyz, idx0=pelvis), fps.  Joints must be in UniMoTok JOINT_NAMES order.

We resample 120->30, build root_pos(m,z-up) + root_quat(wxyz), reorder joints, and write a
minimal body array (idx0 = real root; other bodies = root broadcast / identity -- they are
NOT used for the 41-D features, only the root is).
"""
from __future__ import annotations
import argparse, os, glob
import numpy as np
from scipy.spatial.transform import Rotation as R

SRC_FPS, DST_FPS, STRIDE = 120.0, 30.0, 4

# OMG/seed-CSV sequential joint order
OMG = ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint","waist_yaw_joint","waist_roll_joint","waist_pitch_joint","left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
# UniMoTok JOINT_NAMES order (from stage2/export_g1_motion.py)
UNIMOTOK = ["left_hip_pitch_joint","right_hip_pitch_joint","waist_yaw_joint","left_hip_roll_joint","right_hip_roll_joint","waist_roll_joint","left_hip_yaw_joint","right_hip_yaw_joint","waist_pitch_joint","left_knee_joint","right_knee_joint","left_shoulder_pitch_joint","right_shoulder_pitch_joint","left_ankle_pitch_joint","right_ankle_pitch_joint","left_shoulder_roll_joint","right_shoulder_roll_joint","left_ankle_roll_joint","right_ankle_roll_joint","left_shoulder_yaw_joint","right_shoulder_yaw_joint","left_elbow_joint","right_elbow_joint","left_wrist_roll_joint","right_wrist_roll_joint","left_wrist_pitch_joint","right_wrist_pitch_joint","left_wrist_yaw_joint","right_wrist_yaw_joint"]
PERM = np.array([OMG.index(n) for n in UNIMOTOK])  # seed_joints[:,PERM] -> UniMoTok order


def convert_csv(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1).astype(np.float64)
    if a.ndim == 1 or a.shape[0] < STRIDE * 20:  # skip too-short
        return None
    a = a[::STRIDE]
    root_pos = (a[:, 1:4] / 100.0).astype(np.float32)        # cm -> m, z-up
    quat_xyzw = R.from_euler("XYZ", a[:, 4:7], degrees=True).as_quat().astype(np.float32)
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=1)
    joints = np.deg2rad(a[:, 7:36]).astype(np.float32)        # OMG order
    joints_u = joints[:, PERM]                                # -> UniMoTok order
    T = root_pos.shape[0]
    body_pos = np.broadcast_to(root_pos[:, None, :], (T, 30, 3)).copy()   # idx0 = real root
    body_quat = np.tile(np.array([1, 0, 0, 0], np.float32), (T, 30, 1))   # identity for non-root
    body_quat[:, 0, :] = quat_wxyz                                        # idx0 = real root
    return {"joint_pos": joints_u, "body_pos_w": body_pos.astype(np.float32),
            "body_quat_w": body_quat.astype(np.float32), "fps": np.array([DST_FPS], np.float32)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/scratch/user/yzdong/OMG-Data/raw/bones_seed/g1/csv")
    p.add_argument("--out", default="/scratch/user/yzdong/OMG-Data/raw/bones_seed/artifacts_seed")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    args = p.parse_args()
    csvs = sorted(glob.glob(os.path.join(args.src, "**", "*.csv"), recursive=True))
    if args.limit:
        csvs = csvs[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    kept = 0
    for i, c in enumerate(csvs):
        try:
            m = convert_csv(c)
        except Exception:
            continue
        if m is None:
            continue
        name = os.path.splitext(os.path.basename(c))[0]
        d = os.path.join(args.out, f"{name}:v0")
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "motion.npz"), **m)
        kept += 1
        if kept % 2000 == 0:
            print(f"  {kept} clips", flush=True)
    print(f"DONE: {kept} artifacts -> {args.out}")


if __name__ == "__main__":
    main()
