"""Run HoloMotion on the 4 near-ground clips and render reference|executed videos.

Pairs with the BFM-Zero expert|policy videos so the report can show both trackers
on the same clip. Run in the OMG env (mujoco + onnx + render):
    cd OMG && MUJOCO_GL=egl PYTHONPATH=src:<wbt> .venv-cu128/bin/python \
        <wbt>/stage3_sim2sim/bfmzero_compare/render_holomotion_compare.py
"""
import os
import numpy as np

from stage3_sim2sim.sim2sim import build_qpos36_from_artifact, run_tracker
from stage3_sim2sim.joint_order import qpos36_feature_to_omg
from omg.render.mujoco import render_qpos_comparison_video

ART = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/artifacts_seed_full"
ONNX = ("/scratch/user/yzdong/OMG-models/holomotion_dl/"
        "HoloMotion_motion_tracking_model_v1.3.1/exported/motion_tracking_model.onnx")
OMG_ROOT = "/ws/user/yzdong/src/github/OMG"
OUT = os.path.dirname(os.path.abspath(__file__))

CLIPS = [
    ("clip0", "crouch_idle_right_R_003__A247_M:v0", "deep crouch (idle)"),
    ("clip1", "crouch_ff_start_270_R_001__A197_M:v0", "crouch + turn"),
    ("clip2", "sit_on_chair_stop_R_001__A047:v0", "sit down"),
    ("clip3", "squat_001__A360:v0", "squat"),
]

for cid, art, desc in CLIPS:
    q = build_qpos36_from_artifact(f"{ART}/{art}/motion.npz")     # FEATURE order
    roll = run_tracker(q, 30, f"/tmp/holo_{cid}", ONNX, OMG_ROOT, num_frames=q.shape[0])
    d = np.load(roll)
    ex = d["executed_qpos_36"]                                     # OMG/MuJoCo order
    ref_omg = qpos36_feature_to_omg(q)[:len(ex)]                   # reorder ref for renderer
    mp4 = f"{OUT}/holo_{cid}.mp4"
    render_qpos_comparison_video(ref_omg, ex, mp4, fps=30,
                                 left_title="Reference", right_title="HoloMotion")
    print(f"{cid} {desc}: rendered {mp4}  ({len(ex)} frames)")
print("done")
