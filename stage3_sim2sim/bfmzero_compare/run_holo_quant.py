"""Run HoloMotion on all quant clips; save each rollout to quant/holo_<idx>.npz.

Runs on CPU ONNX (run_tracker sets CUDA_VISIBLE_DEVICES=""), so it can run in
parallel with the BFM-Zero GPU sweep. Run in the OMG env.
"""
import os
import numpy as np
from stage3_sim2sim.sim2sim import build_qpos36_from_artifact, run_tracker
from stage3_sim2sim.bfmzero_compare.quant_clips import QUANT_CLIPS, ART

ONNX = ("/scratch/user/yzdong/OMG-models/holomotion_dl/"
        "HoloMotion_motion_tracking_model_v1.3.1/exported/motion_tracking_model.onnx")
OMG_ROOT = "/ws/user/yzdong/src/github/OMG"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant")

for idx, art, cat, label in QUANT_CLIPS:
    out = f"{DST}/holo_{idx}.npz"
    if os.path.exists(out):
        print(f"[{idx}] {cat} exists, skip"); continue
    q = build_qpos36_from_artifact(f"{ART}/{art}/motion.npz")   # FEATURE order
    roll = run_tracker(q, 30, f"/tmp/holoq_{idx}", ONNX, OMG_ROOT, num_frames=q.shape[0])
    d = np.load(roll)
    np.savez(out, executed_qpos_36=d["executed_qpos_36"], reference_qpos_36=d["reference_qpos_36"])
    print(f"[{idx}] {cat:6s} {label}: holo rollout saved ({d['executed_qpos_36'].shape[0]} frames)")
print("HOLO QUANT DONE")
