"""Convert our BONES-SEED G1 artifact clips -> a BFM-Zero motion .pkl.

BFM-Zero's motion-lib (HumanoidVerse `Humanoid_Batch`) is a *robot* forward-kinematics
loader (NOT SMPL mesh FK, despite the "mesh_parser" name). It needs only:
    root_trans_offset [T,3]  (z-up pelvis position, world)
    pose_aa           [T,30,3] per-joint axis-angle: row 0 = root rotvec,
                       rows 1..29 = dof[j] * joint_axis (G1 XML axis, all unit +)
    fps               int
From these it derives dof_pos (= pose_aa[:,1:].sum(-1)), velocities and body FK.

Our artifact joints are in FEATURE order (HoloMotion convention); BFM-Zero's G1 XML
is in OMG/MuJoCo order -> reorder with qpos36_feature_to_omg before building pose_aa.
This lets us run our EXACT crouch/sit/squat clips through BFM-Zero for an
apples-to-apples comparison against HoloMotion.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as R

from stage3_sim2sim.sim2sim import build_qpos36_from_artifact
from stage3_sim2sim.joint_order import qpos36_feature_to_omg

# G1 29-dof joint axes in BFM-Zero XML (OMG) order: left leg, right leg, waist, L arm, R arm.
# Parsed from g1_29dof_old_freebase_noadditional_actuators.xml (all unit +1 axes).
G1_DOF_AXIS = np.array([
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],   # left leg
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],   # right leg
    [0, 0, 1], [1, 0, 0], [0, 1, 0],                                     # waist yaw/roll/pitch
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # left arm
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # right arm
], dtype=np.float64)


def qpos36_omg_to_bfmzero_motion(qpos36_omg: np.ndarray, fps: int = 30) -> dict:
    """qpos_36 (joints already in OMG order) -> BFM-Zero motion entry dict."""
    q = np.asarray(qpos36_omg, dtype=np.float64)
    T = q.shape[0]
    trans = q[:, 0:3].copy()                       # z-up pelvis
    quat_wxyz = q[:, 3:7]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    dof = q[:, 7:36]                                # 29, OMG order

    pose_aa = np.zeros((T, 30, 3), dtype=np.float32)
    pose_aa[:, 0, :] = R.from_quat(quat_xyzw).as_rotvec()      # root
    pose_aa[:, 1:30, :] = (dof[:, :, None] * G1_DOF_AXIS[None]).astype(np.float32)

    return {
        "root_trans_offset": trans.astype(np.float32),
        "pose_aa": pose_aa,
        "dof": dof.astype(np.float32),
        "root_rot": quat_xyzw.astype(np.float32),   # xyzw (unused by loader, kept for parity)
        "fps": int(fps),
    }


def artifact_to_motion(artifact_dir: str, fps: int = 30) -> tuple[str, dict]:
    npz = Path(artifact_dir) / "motion.npz"
    qpos_feat = build_qpos36_from_artifact(str(npz))     # joints in FEATURE order
    qpos_omg = qpos36_feature_to_omg(qpos_feat)          # -> OMG order for BFM-Zero XML
    name = Path(artifact_dir).name.replace(":", "_")
    return name, qpos36_omg_to_bfmzero_motion(qpos_omg, fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", nargs="+", required=True, help="artifact dirs (each has motion.npz)")
    ap.add_argument("--out", required=True, help="output .pkl path")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    out = {}
    order = []
    for d in a.artifacts:
        name, motion = artifact_to_motion(d, fps=a.fps)
        out[name] = motion
        order.append(name)
        q = motion["pose_aa"]
        print(f"  [{len(order)-1}] {name}: T={q.shape[0]} "
              f"root_z[min={motion['root_trans_offset'][:,2].min():.3f} "
              f"max={motion['root_trans_offset'][:,2].max():.3f}]")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, a.out)
    print(f"\nwrote {len(out)} clips -> {a.out}")
    print("motion-id order:", {i: n for i, n in enumerate(order)})


if __name__ == "__main__":
    main()
