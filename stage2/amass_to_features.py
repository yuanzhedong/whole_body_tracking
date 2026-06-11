"""Convert pre-retargeted-to-G1 AMASS clips (HF fleaven/Retargeted_AMASS_for_robotics, g1/*.npy)
into 41-D G1 motion features matching our LAFAN pipeline — pure numpy, no Isaac.

HF .npy [N,36]: cols 0:3 root pos (z offset by -0.793), 3:7 root quat xyzw, 7:36 = 29 joints in
csv_to_npz BLOCK order. Our 41-D features use the robot INTERLEAVED joint order, so we REMAP
block->interleaved (csv_to_npz feeds block names to the articulation, which reorders to interleaved
in the saved npz). Then build_features (root6d + local lin/ang vel + joints, y-up) at 20fps.

Usage: .venv/bin/python stage2/amass_to_features.py --files g1/ACCAD/.../*.npy --out_dir <dir>
"""
import argparse, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_g1_motion import build_features  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402

# csv_to_npz BLOCK order (input order of the HF .npy joints)
BLOCK = ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint",
    "left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint",
    "right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint","left_shoulder_pitch_joint",
    "left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint",
    "left_wrist_pitch_joint","left_wrist_yaw_joint","right_shoulder_pitch_joint",
    "right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint",
    "right_wrist_pitch_joint","right_wrist_yaw_joint"]
# robot INTERLEAVED order (our 41-D feature joint order)
FEAT = ["left_hip_pitch_joint","right_hip_pitch_joint","waist_yaw_joint","left_hip_roll_joint",
    "right_hip_roll_joint","waist_roll_joint","left_hip_yaw_joint","right_hip_yaw_joint",
    "waist_pitch_joint","left_knee_joint","right_knee_joint","left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint","left_ankle_pitch_joint","right_ankle_pitch_joint",
    "left_shoulder_roll_joint","right_shoulder_roll_joint","left_ankle_roll_joint",
    "right_ankle_roll_joint","left_shoulder_yaw_joint","right_shoulder_yaw_joint","left_elbow_joint",
    "right_elbow_joint","left_wrist_roll_joint","right_wrist_roll_joint","left_wrist_pitch_joint",
    "right_wrist_pitch_joint","left_wrist_yaw_joint","right_wrist_yaw_joint"]
PERM = [BLOCK.index(f) for f in FEAT]   # joints_interleaved = joints_block[:, PERM]
Z_OFF = 0.793


def convert_npy(npy_path, target_fps=20):
    m = re.search(r"_(\d+)_jpos", os.path.basename(npy_path))
    src_fps = int(m.group(1)) if m else 30
    d = np.load(npy_path).astype(np.float64)
    if d.shape[1] < 36 or len(d) < 4:
        return None
    pos = d[:, 0:3].copy(); pos[:, 2] += Z_OFF
    quat = d[:, 3:7]                                   # xyzw
    joints = d[:, 7:36][:, PERM]                       # block -> interleaved
    N = len(d); T = max(2, round(N / src_fps * target_fps))
    ti = np.linspace(0, 1, N); to = np.linspace(0, 1, T)
    pos_r = np.stack([np.interp(to, ti, pos[:, c]) for c in range(3)], 1)
    joints_r = np.stack([np.interp(to, ti, joints[:, c]) for c in range(29)], 1)
    quat_r = Slerp(ti, Rotation.from_quat(quat))(to).as_quat()
    feat = build_features(pos_r, quat_r, joints_r, dt=1.0 / target_fps, to_yup=True)
    if not np.isfinite(feat).all():
        return None
    return feat.astype(np.float32)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True, help="local paths to HF g1/*.npy")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", default="amass")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for f in args.files:
        feat = convert_npy(f)
        if feat is None:
            print("skip (bad):", f); continue
        name = args.prefix + "_" + os.path.basename(f).replace(".npy", "").replace("/", "_")[:60]
        np.savez(os.path.join(args.out_dir, name + ".npz"), motion=feat)
        n += 1
    print(f"converted {n} clips -> {args.out_dir}")
