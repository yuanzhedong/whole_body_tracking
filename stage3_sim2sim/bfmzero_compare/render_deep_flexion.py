"""Deep-flexion demos: Reference | HoloMotion | BFM-Zero with a live knee-angle readout.

Visualizes the root cause: as the posture deepens, the reference knee bends to ~150deg
and BFM-Zero follows while HoloMotion stalls (~80deg). Each panel shows the robot AND
its live left-knee flexion. Joint order per source: the HoloMotion rollout executed is
FEATURE order (reorder to OMG for the renderer); BFM-Zero is OMG. Run in the OMG env.
"""
import os
import subprocess
import numpy as np

from stage3_sim2sim.sim2sim import build_qpos36_from_artifact
from stage3_sim2sim.joint_order import qpos36_feature_to_omg, OMG_ORDER
from omg.render.mujoco import render_qpos_video

HERE = os.path.dirname(os.path.abspath(__file__))
ART = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/artifacts_seed_full"
FFMPEG = "/usr/bin/ffmpeg"
LK = 7 + OMG_ORDER.index("left_knee_joint")
# (name, artifact, quant index) -- quant/holo_<i>.npz (FEATURE) + quant/bfm_<i>.npz (OMG)
CLIPS = [
    ("squat", "squat_001__A360:v0", 1),
    ("crouch", "crouch_ff_start_180_R_003__A145_M:v0", 0),
]


def resample(a, n):
    return a[np.round(np.linspace(0, len(a) - 1, n)).astype(int)]


def knee_lines(q):
    return [[f"L-knee flexion: {abs(np.degrees(q[f, LK])):3.0f} deg"] for f in range(len(q))]


for name, clip, idx in CLIPS:
    ref = qpos36_feature_to_omg(build_qpos36_from_artifact(f"{ART}/{clip}/motion.npz"))
    n = len(ref)
    holo = qpos36_feature_to_omg(resample(np.load(f"{HERE}/quant/holo_{idx}.npz")["executed_qpos_36"], n))
    bfm = resample(np.load(f"{HERE}/quant/bfm_{idx}.npz")["executed_qpos_36"], n)

    panels = []
    for tag, q, title in [("ref", ref, "Reference"), ("holo", holo, "HoloMotion"), ("bfm", bfm, "BFM-Zero")]:
        p = f"{HERE}/_df_{name}_{tag}.mp4"
        render_qpos_video(q[:n], p, fps=30, width=640, height=720, title=title,
                          per_frame_info_lines=knee_lines(q[:n]))
        panels.append(p)
    out = f"{HERE}/deep_flexion_{name}.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", panels[0], "-i", panels[1], "-i", panels[2],
                    "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out], check=True)
    for p in panels:
        os.remove(p)
    print(f"{name}: {out} ({n} frames)  peak |L-knee| ref {abs(np.degrees(ref[:, LK])).max():.0f} "
          f"holo {abs(np.degrees(holo[:, LK])).max():.0f} bfm {abs(np.degrees(bfm[:, LK])).max():.0f} deg")
