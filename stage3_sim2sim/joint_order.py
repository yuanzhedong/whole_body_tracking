"""G1 joint-order conventions and the mapping between them.

There are TWO 29-DOF orderings in play and they are NOT the same:

* **FEATURE / artifact / VAE order** ("interleaved") — used by the seed adapter,
  ``export_g1_motion`` features, the VAE, and the qpos_36 those produce. Layout
  interleaves left/right/waist (lhp, rhp, waist_yaw, lhr, ...).
* **OMG / MuJoCo / renderer order** ("sequential") — used by ``G1Kinematics``,
  ``render_qpos_frames`` and the MuJoCo ``g1_29dof`` model (qpos). Full left leg,
  full right leg, waist, left arm, right arm.

The VAE and the HoloMotion tracker operate consistently in the FEATURE order, so
training and sim2sim are unaffected by this. But to **render** a feature-order
qpos with the OMG/MuJoCo renderer you must first reorder the 29 joints into OMG
order via ``feature_to_omg`` (otherwise the pose is scrambled — the cause of the
"weird walking" in early kinematic demo videos).
"""
from __future__ import annotations
import numpy as np

# feature/artifact/VAE order (matches stage2/export_g1_motion.JOINT_NAMES)
FEATURE_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# OMG / MuJoCo g1_29dof / renderer order (matches omg.robots.g1.constants.G1_JOINT_NAMES)
OMG_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# index arrays: out[:, i] = inp[:, PERM[i]]
FEATURE_TO_OMG = np.array([FEATURE_ORDER.index(n) for n in OMG_ORDER])
OMG_TO_FEATURE = np.array([OMG_ORDER.index(n) for n in FEATURE_ORDER])


def feature_to_omg(joints29):
    """Reorder [...,29] joints from FEATURE order to OMG/MuJoCo/renderer order."""
    return np.asarray(joints29)[..., FEATURE_TO_OMG]


def qpos36_feature_to_omg(qpos36):
    """qpos_36 [...,36]=[root3,quat4,29 joints] -> joints reordered to OMG/render order."""
    q = np.array(qpos36, copy=True)
    q[..., 7:36] = np.asarray(qpos36)[..., 7:36][..., FEATURE_TO_OMG]
    return q
