"""Canonical G1 body (link) ordering as Isaac/Isaac-Lab enumerates it for this URDF.

`scripts/csv_to_npz.py` writes body_pos_w/body_quat_w in this order but does NOT store the
names, so assess_motion.py / viz_motion.py reference this constant. Verified against the
rollout logger's `robot.data.body_names` for the same G1 model (30 bodies)."""

G1_BODY_NAMES = [
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link", "left_hip_yaw_link",
    "right_hip_yaw_link", "torso_link", "left_knee_link", "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_ankle_pitch_link",
    "right_ankle_pitch_link", "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link", "left_shoulder_yaw_link",
    "right_shoulder_yaw_link", "left_elbow_link", "right_elbow_link", "left_wrist_roll_link",
    "right_wrist_roll_link", "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]
PELVIS_IDX = G1_BODY_NAMES.index("pelvis")
FOOT_IDX = [G1_BODY_NAMES.index("left_ankle_roll_link"),
            G1_BODY_NAMES.index("right_ankle_roll_link")]
