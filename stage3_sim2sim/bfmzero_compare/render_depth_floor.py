"""Depth-floor demo: Reference | BFM-Zero on a floor-sitting clip, with a live pelvis
height readout. Visualizes BFM-Zero's genuine ~0.35-0.40 m pelvis floor: the
reference sits at ~0.1 m but BFM-Zero stays ~0.5 m (stable, but doesn't reach the
floor). Run in the OMG env.
"""
import os
import subprocess
import numpy as np

from stage3_sim2sim.sim2sim import build_qpos36_from_artifact
from stage3_sim2sim.joint_order import qpos36_feature_to_omg
from stage3_sim2sim.bfmzero_compare.build_large_sample import ARTROOT
from omg.render.mujoco import render_qpos_video

HERE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = "/usr/bin/ffmpeg"
CLIP = "sitting_legs_cross_arm_side_loop_003__A046_M:v0"
IDX = 118


def resample(a, n):
    return a[np.round(np.linspace(0, len(a) - 1, n)).astype(int)]


def pelvis_lines(q):
    return [[f"pelvis: {q[f, 2]:.2f} m"] for f in range(len(q))]


ref = qpos36_feature_to_omg(build_qpos36_from_artifact(f"{ARTROOT}/{CLIP}/motion.npz"))
n = len(ref)
bfm = resample(np.load(f"{HERE}/large/bfm/rollout_{IDX}.npz")["executed_qpos_36"], n)  # OMG order

panels = []
for tag, q, title in [("ref", ref, "Reference (floor sit)"), ("bfm", bfm, "BFM-Zero")]:
    p = f"{HERE}/_dfl_{tag}.mp4"
    render_qpos_video(q[:n], p, fps=30, width=640, height=720, title=title,
                      per_frame_info_lines=pelvis_lines(q[:n]))
    panels.append(p)
out = f"{HERE}/depth_floor.mp4"
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", panels[0], "-i", panels[1],
                "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out], check=True)
for p in panels:
    os.remove(p)
print(f"wrote {out}  ref pelvis min {ref[:, 2].min():.2f}m  BFM pelvis min {bfm[:, 2].min():.2f}m")
