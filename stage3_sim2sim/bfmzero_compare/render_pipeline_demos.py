"""Pipeline demo videos for the VAE->BFM-Zero report.

For each chosen quant clip: [ Reference | BFM-Zero on original | BFM-Zero on
VAE-decoded ] -> demo_<idx>.mp4. Shows BFM-Zero executes the VAE-reconstructed
motion like the original across crouch / squat / sit. Run in the OMG env.
"""
import os
import subprocess
import sys
import numpy as np

from stage3_sim2sim.sim2sim import build_qpos36_from_artifact
from stage3_sim2sim.joint_order import qpos36_feature_to_omg
from stage3_sim2sim.bfmzero_compare.quant_clips import QUANT_CLIPS, ART
from omg.render.mujoco import render_qpos_video

HERE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = "/usr/bin/ffmpeg"
# clip indices to render (must have quant/bfm_<i>.npz and quant/dec_<i>.npz)
INDICES = [int(x) for x in sys.argv[1:]] or [1, 3, 5, 6]
BY_IDX = {i: (a, c, lab) for i, a, c, lab in QUANT_CLIPS}


def resample(a, n):
    return a[np.round(np.linspace(0, len(a) - 1, n)).astype(int)]


for idx in INDICES:
    art, cat, label = BY_IDX[idx]
    b, d = f"{HERE}/quant/bfm_{idx}.npz", f"{HERE}/quant/dec_{idx}.npz"
    if not (os.path.exists(b) and os.path.exists(d)):
        print(f"[{idx}] missing rollout(s), skip"); continue
    ref = qpos36_feature_to_omg(build_qpos36_from_artifact(f"{ART}/{art}/motion.npz"))
    n = len(ref)
    bfm_o = resample(np.load(b)["executed_qpos_36"], n)
    bfm_d = resample(np.load(d)["executed_qpos_36"], n)
    panels = []
    for tag, q, title in [("ref", ref, "Reference"),
                          ("o", bfm_o, "BFM-Zero: original"),
                          ("d", bfm_d, "BFM-Zero: VAE-decoded")]:
        p = f"{HERE}/_d{idx}_{tag}.mp4"
        render_qpos_video(q[:n], p, fps=30, width=640, height=720, title=title)
        panels.append(p)
    out = f"{HERE}/demo_{idx}.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", panels[0], "-i", panels[1], "-i", panels[2],
                    "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out], check=True)
    for p in panels:
        os.remove(p)
    print(f"[{idx}] {cat:6s} {label}: {out} ({n} frames)")
print("done")
