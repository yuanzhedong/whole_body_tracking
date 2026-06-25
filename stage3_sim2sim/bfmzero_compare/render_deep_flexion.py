"""Deep-flexion demo: Reference | HoloMotion | BFM-Zero with a live knee-angle readout.

Visualizes the root cause: as the squat deepens, the reference knee bends to ~150deg
and BFM-Zero follows (~130deg) while HoloMotion stalls (~80deg). Each panel shows the
robot AND its live left-knee flexion. Correct joint order per source: the HoloMotion
rollout executed is FEATURE order (reorder to OMG for the renderer); BFM-Zero is OMG.
Run in the OMG env.
"""
import os
import subprocess
import numpy as np

from stage3_sim2sim.sim2sim import build_qpos36_from_artifact
from stage3_sim2sim.joint_order import qpos36_feature_to_omg, OMG_ORDER
from omg.render.mujoco import render_qpos_video

HERE = os.path.dirname(os.path.abspath(__file__))
ART = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/artifacts_seed_full"
CLIP = "squat_001__A360:v0"          # quant idx 1
FFMPEG = "/usr/bin/ffmpeg"
LK = 7 + OMG_ORDER.index("left_knee_joint")


def resample(a, n):
    return a[np.round(np.linspace(0, len(a) - 1, n)).astype(int)]


ref = qpos36_feature_to_omg(build_qpos36_from_artifact(f"{ART}/{CLIP}/motion.npz"))
n = len(ref)
holo = qpos36_feature_to_omg(resample(np.load(f"{HERE}/quant/holo_1.npz")["executed_qpos_36"], n))
bfm = resample(np.load(f"{HERE}/quant/bfm_1.npz")["executed_qpos_36"], n)          # already OMG


def knee_lines(q):
    return [[f"L-knee flexion: {abs(np.degrees(q[f, LK])):3.0f} deg"] for f in range(len(q))]


panels = []
for tag, q, title in [("ref", ref, "Reference"), ("holo", holo, "HoloMotion"), ("bfm", bfm, "BFM-Zero")]:
    p = f"{HERE}/_df_{tag}.mp4"
    render_qpos_video(q[:n], p, fps=30, width=640, height=720, title=title,
                      per_frame_info_lines=knee_lines(q[:n]))
    panels.append(p)

out = f"{HERE}/deep_flexion_squat.mp4"
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", panels[0], "-i", panels[1], "-i", panels[2],
                "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out], check=True)
for p in panels:
    os.remove(p)
# peak flexion summary
print(f"wrote {out} ({n} frames)")
print(f"max |L-knee|:  reference {abs(np.degrees(ref[:, LK])).max():.0f}deg  "
      f"HoloMotion {abs(np.degrees(holo[:, LK])).max():.0f}deg  BFM-Zero {abs(np.degrees(bfm[:, LK])).max():.0f}deg")
