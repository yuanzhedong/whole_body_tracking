"""Mirror-augment a G1 41-D motion dataset (UniMoTok trick: free L/R-symmetry data doubling).
Builds <src>_mir with each clip + its sagittal mirror. Verified: mirror(mirror(x))==x.

G1 41-D feature = root6d[0:6] + root_lin_vel[6:9] + root_ang_vel[9:12] + joints[12:41].
Joints are stored INTERLEAVED (left,right,waist,...), NOT L-block/R-block — see NAMES.
Mirror: swap left<->right joint, flip sign on roll/yaw DOF (waist_pitch/knee/elbow/pitch unchanged);
root: y-up (x=fwd,y=up,z=right) sagittal reflect (flip z) -> 6d cols' z [2,5], lin_vel z [8],
ang_vel x,y [9,10] (pseudovector).

Usage: .venv/bin/python stage2/mirror_augment.py <src_dataset_dir>
"""
import numpy as np, os, sys

NAMES = ["left_hip_pitch_joint","right_hip_pitch_joint","waist_yaw_joint","left_hip_roll_joint",
    "right_hip_roll_joint","waist_roll_joint","left_hip_yaw_joint","right_hip_yaw_joint",
    "waist_pitch_joint","left_knee_joint","right_knee_joint","left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint","left_ankle_pitch_joint","right_ankle_pitch_joint",
    "left_shoulder_roll_joint","right_shoulder_roll_joint","left_ankle_roll_joint",
    "right_ankle_roll_joint","left_shoulder_yaw_joint","right_shoulder_yaw_joint","left_elbow_joint",
    "right_elbow_joint","left_wrist_roll_joint","right_wrist_roll_joint","left_wrist_pitch_joint",
    "right_wrist_pitch_joint","left_wrist_yaw_joint","right_wrist_yaw_joint"]

_jperm = list(range(29)); _jsign = np.ones(29)
for i, n in enumerate(NAMES):
    p = ("right_"+n[5:]) if n.startswith("left_") else ("left_"+n[6:]) if n.startswith("right_") else n
    _jperm[i] = NAMES.index(p)
    _jsign[i] = -1.0 if ("roll" in n or "yaw" in n) else 1.0
_rsign = np.ones(12); _rsign[[2, 5, 8, 9, 10]] = -1.0


def mirror(feat):  # feat [T,41] -> mirrored [T,41]
    out = feat.copy()
    out[:, :12] = feat[:, :12] * _rsign
    out[:, 12:] = feat[:, 12:][:, _jperm] * _jsign
    return out


if __name__ == "__main__":
    src = sys.argv[1].rstrip("/"); dst = src + "_mir"
    assert np.abs(mirror(mirror(np.random.randn(64, 41))) - np.random.randn(64, 41)).size  # noqa
    x = np.random.randn(64, 41)
    assert np.abs(mirror(mirror(x)) - x).max() < 1e-9, "double-mirror != identity"
    os.system(f"rm -rf {dst}; mkdir -p {dst}/train {dst}/val; cp {src}/normalization.npz {dst}/")
    n = 0
    for sp in ["train", "val"]:
        d = f"{src}/{sp}"
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz"): continue
            o = np.load(f"{d}/{f}"); m = dict(o)
            np.savez(f"{dst}/{sp}/{f}", **m)
            m2 = dict(o); m2["motion"] = mirror(o["motion"].astype(np.float32)).astype(np.float32)
            np.savez(f"{dst}/{sp}/{f[:-4]}_mir.npz", **m2); n += 2
    print(f"built {dst}: {n} files (double-mirror verified)")
