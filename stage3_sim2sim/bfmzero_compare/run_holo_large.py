"""Run HoloMotion on the large sample -> large/holo/holo_<idx>.npz (CPU, parallel-safe)."""
import json
import os
import numpy as np
from stage3_sim2sim.sim2sim import build_qpos36_from_artifact, run_tracker
from stage3_sim2sim.bfmzero_compare.build_large_sample import ARTROOT

ONNX = ("/scratch/user/yzdong/OMG-models/holomotion_dl/"
        "HoloMotion_motion_tracking_model_v1.3.1/exported/motion_tracking_model.onnx")
OMG_ROOT = "/ws/user/yzdong/src/github/OMG"
HERE = os.path.dirname(os.path.abspath(__file__))
DST = f"{HERE}/large/holo"

import sys
man = json.load(open(f"{HERE}/large_sample.json"))
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end = int(sys.argv[2]) if len(sys.argv) > 2 else len(man)
done = 0
for m in man[start:end]:
    out = f"{DST}/holo_{m['idx']}.npz"
    if os.path.exists(out):
        done += 1
        continue
    try:
        q = build_qpos36_from_artifact(f"{ARTROOT}/{m['artifact']}/motion.npz")
        roll = run_tracker(q, 30, f"/tmp/hl_{m['idx']}", ONNX, OMG_ROOT, num_frames=q.shape[0], timeout=600)
        d = np.load(roll)
        np.savez(out, executed_qpos_36=d["executed_qpos_36"], reference_qpos_36=d["reference_qpos_36"])
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(man)} done", flush=True)
    except Exception as e:
        print(f"  motion {m['idx']} ({m['artifact'][:30]}) FAILED: {repr(e)[:100]}", flush=True)
print(f"HOLO LARGE DONE: {done}/{len(man)}")
