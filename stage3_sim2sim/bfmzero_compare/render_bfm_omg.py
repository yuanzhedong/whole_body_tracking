"""Re-render the BFM-Zero rollouts with the SAME OMG renderer as HoloMotion.

So both trackers' videos share one style/size and can be stacked per clip.
BFM-Zero ran at 50 Hz; resample each rollout to the matching HoloMotion frame
count (30 Hz) so the stacked panels stay time-aligned. Run in the OMG env.
"""
import os
import glob
import numpy as np
from omg.render.mujoco import render_qpos_comparison_video

HERE = os.path.dirname(os.path.abspath(__file__))
# (cid, holo frame count) — match so vstack pairs frames 1:1
HOLO_N = {"clip0": 286, "clip1": 247, "clip2": 160, "clip3": 290}


def resample(a, n):
    idx = np.round(np.linspace(0, len(a) - 1, n)).astype(int)
    return a[idx]


for cid, n in HOLO_N.items():
    d = np.load(f"{HERE}/rollout_{cid}.npz")
    ex = resample(d["executed_qpos_36"], n)
    ref = resample(d["reference_qpos_36"], n)
    mp4 = f"{HERE}/bfm_omg_{cid}.mp4"
    render_qpos_comparison_video(ref, ex, mp4, fps=30,
                                 left_title="Reference", right_title="BFM-Zero")
    print(f"{cid}: rendered {mp4} ({n} frames)")
print("done")
