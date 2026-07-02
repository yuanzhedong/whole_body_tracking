"""Run HoloMotion on the seed-sample clips -> seed/holo_<idx>.npz (CPU, parallel-safe)."""
import json
import os
import numpy as np
from stage3_sim2sim.sim2sim import build_qpos36_from_artifact, run_tracker

HERE = os.path.dirname(os.path.abspath(__file__))
ART = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/artifacts_seed_full"
ONNX = ("/scratch/user/yzdong/OMG-models/holomotion_dl/"
        "HoloMotion_motion_tracking_model_v1.3.1/exported/motion_tracking_model.onnx")
OMG_ROOT = "/ws/user/yzdong/src/github/OMG"

manifest = json.load(open(f"{HERE}/seed_sample.json"))
for m in manifest:
    idx = m["idx"]
    out = f"{HERE}/seed/holo_{idx}.npz"
    if os.path.exists(out):
        print(f"[{idx}] exists, skip"); continue
    q = build_qpos36_from_artifact(f"{ART}/{m['artifact']}/motion.npz")
    roll = run_tracker(q, 30, f"/tmp/holos_{idx}", ONNX, OMG_ROOT, num_frames=q.shape[0])
    d = np.load(roll)
    np.savez(out, executed_qpos_36=d["executed_qpos_36"], reference_qpos_36=d["reference_qpos_36"])
    print(f"[{idx}] {m['artifact'][:40]}: holo saved ({d['executed_qpos_36'].shape[0]}f)")
print("HOLO SEED DONE")
