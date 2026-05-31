"""Convert a pre-retargeted-to-G1 AMASS clip (HuggingFace `fleaven/Retargeted_AMASS_for_robotics`)
into the CSV format that this repo's `scripts/csv_to_npz.py` consumes.

The HF G1 `.npy` arrays are shape [N, 36]:
    cols 0:3   root world position (x, y, z)   -- z is offset by -0.793 (nominal pelvis height)
    cols 3:7   root quaternion, order xyzw
    cols 7:36  29 G1 joint angles, in EXACTLY this repo's joint order (csv_to_npz.py:331-361)
fps is encoded in the filename, e.g. `*_poses_120_jpos.npy` -> 120, `*_poses_100_jpos.npy` -> 100.

So conversion is: download -> add 0.793 to root z (per the dataset's own visualize.py) -> validate
-> write a headerless CSV. The column layout already matches csv_to_npz.py's contract and the quat
is already xyzw (csv_to_npz reorders [3,0,1,2] internally), so no remap is needed.

Usage:
    python retargeting/hf_to_csv.py --list-walks            # browse available walk clips
    python retargeting/hf_to_csv.py \
        --file "g1/ACCAD/s007/QkWalk1_poses_120_jpos.npy" \
        --out retargeting/out/amass_accad_qkwalk1.csv
Then feed the printed --input_fps to csv_to_npz.py.
"""
import argparse
import json
import os
import re
import numpy as np

REPO = "fleaven/Retargeted_AMASS_for_robotics"
PELVIS_HEIGHT_OFFSET = 0.793  # from the dataset's g1/visualize.py read_rtj()

# 29 G1 joints in csv_to_npz.py order, with URDF limits (rad) for a sanity check.
JOINT_LIMITS = [
    ("left_hip_pitch_joint", -2.5307, 2.8798), ("left_hip_roll_joint", -0.5236, 2.9671),
    ("left_hip_yaw_joint", -2.7576, 2.7576), ("left_knee_joint", -0.087267, 2.8798),
    ("left_ankle_pitch_joint", -0.87267, 0.5236), ("left_ankle_roll_joint", -0.2618, 0.2618),
    ("right_hip_pitch_joint", -2.5307, 2.8798), ("right_hip_roll_joint", -2.9671, 0.5236),
    ("right_hip_yaw_joint", -2.7576, 2.7576), ("right_knee_joint", -0.087267, 2.8798),
    ("right_ankle_pitch_joint", -0.87267, 0.5236), ("right_ankle_roll_joint", -0.2618, 0.2618),
    ("waist_yaw_joint", -2.618, 2.618), ("waist_roll_joint", -0.52, 0.52),
    ("waist_pitch_joint", -0.52, 0.52), ("left_shoulder_pitch_joint", -3.0892, 2.6704),
    ("left_shoulder_roll_joint", -1.5882, 2.2515), ("left_shoulder_yaw_joint", -2.618, 2.618),
    ("left_elbow_joint", -1.0472, 2.0944), ("left_wrist_roll_joint", -1.972222, 1.972222),
    ("left_wrist_pitch_joint", -1.614430, 1.614430), ("left_wrist_yaw_joint", -1.614430, 1.614430),
    ("right_shoulder_pitch_joint", -3.0892, 2.6704), ("right_shoulder_roll_joint", -2.2515, 1.5882),
    ("right_shoulder_yaw_joint", -2.618, 2.618), ("right_elbow_joint", -1.0472, 2.0944),
    ("right_wrist_roll_joint", -1.972222, 1.972222), ("right_wrist_pitch_joint", -1.614430, 1.614430),
    ("right_wrist_yaw_joint", -1.614430, 1.614430),
]
N_COLS = 36  # 3 root pos + 4 quat + 29 joints


def parse_fps_from_filename(path: str) -> int:
    """`.../walking_01_poses_100_jpos.npy` -> 100 ; `..._poses_120_jpos.npy` -> 120."""
    m = re.search(r"_poses_(\d+)_jpos", os.path.basename(path))
    if not m:
        m = re.search(r"_(\d+)_jpos", os.path.basename(path))
    if not m:
        raise ValueError(f"could not parse fps from filename: {path}")
    return int(m.group(1))


def apply_height_offset(arr: np.ndarray, offset: float = PELVIS_HEIGHT_OFFSET) -> np.ndarray:
    """Add the nominal pelvis height to root z (col 2), matching the dataset's visualize.py."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out[:, 2] += offset
    return out


def validate(arr: np.ndarray) -> dict:
    """Return a report dict; raises on hard format errors (shape/NaN)."""
    if arr.ndim != 2 or arr.shape[1] != N_COLS:
        raise ValueError(f"expected [N,{N_COLS}], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("array contains non-finite values")
    quat = arr[:, 3:7]
    quat_norm = np.linalg.norm(quat, axis=1)
    joints = arr[:, 7:]
    lows = np.array([lo for _, lo, _ in JOINT_LIMITS])
    highs = np.array([hi for _, _, hi in JOINT_LIMITS])
    below = joints < lows[None, :]
    above = joints > highs[None, :]
    viol = below | above
    per_joint_viol = viol.mean(axis=0)  # fraction of frames each joint is out of range
    worst = sorted(
        [(JOINT_LIMITS[i][0], float(per_joint_viol[i])) for i in range(29)],
        key=lambda x: -x[1],
    )[:5]
    return {
        "frames": int(arr.shape[0]),
        "root_z_min": float(arr[:, 2].min()),
        "root_z_max": float(arr[:, 2].max()),
        "root_xy_travel_m": float(np.linalg.norm(arr[-1, :2] - arr[0, :2])),
        "quat_norm_mean": float(quat_norm.mean()),
        "quat_norm_max_dev": float(np.abs(quat_norm - 1.0).max()),
        "joint_limit_viol_frac_any": float(viol.any(axis=1).mean()),
        "worst_joints": worst,
    }


def write_csv(arr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savetxt(path, arr, delimiter=",", fmt="%.8f")


def download(file_in_repo: str, local_dir: str = "retargeting/data") -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, file_in_repo, repo_type="dataset", local_dir=local_dir)


def list_walks(limit: int = 40):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    walks = sorted(f for f in files if f.startswith("g1/") and "walk" in f.lower())
    for f in walks[:limit]:
        print(f"  {parse_fps_from_filename(f):>3}fps  {f}")
    print(f"... {len(walks)} walk clips total")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", help="path of the .npy inside the HF dataset repo (g1/...)")
    p.add_argument("--out", help="output CSV path")
    p.add_argument("--height_offset", type=float, default=PELVIS_HEIGHT_OFFSET)
    p.add_argument("--list-walks", action="store_true")
    args = p.parse_args()

    if args.list_walks:
        list_walks()
        return
    if not args.file or not args.out:
        p.error("--file and --out are required (or use --list-walks)")

    fps = parse_fps_from_filename(args.file)
    local = download(args.file)
    raw = np.load(local)
    arr = apply_height_offset(raw, args.height_offset)
    report = validate(arr)
    write_csv(arr, args.out)
    report["fps"] = fps
    report["duration_s"] = round(report["frames"] / fps, 2)
    report["out"] = args.out
    report["source"] = f"{REPO}/{args.file}"
    print(json.dumps(report, indent=2))
    print(f"\nNext: ./run.sh scripts/csv_to_npz.py --input_file {args.out} "
          f"--input_fps {fps} --output_name amass_<name> --headless")


if __name__ == "__main__":
    main()
