"""Deep-crouch triptych for the VAE->BFM-Zero pipeline report.

[ Reference | BFM-Zero on original | BFM-Zero on VAE-decoded ] — shows BFM-Zero
executes the VAE-reconstructed deep crouch just like the original (Result 1).
Run in the OMG env.
"""
import os
import subprocess
import numpy as np

from stage3_sim2sim.sim2sim import build_qpos36_from_artifact
from stage3_sim2sim.joint_order import qpos36_feature_to_omg
from omg.render.mujoco import render_qpos_video

HERE = os.path.dirname(os.path.abspath(__file__))
ART = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/artifacts_seed_full"
CROUCH = "crouch_ff_start_180_R_003__A145_M:v0"   # clip0 (grounded deep crouch)
FFMPEG = "/usr/bin/ffmpeg"


def resample(a, n):
    return a[np.round(np.linspace(0, len(a) - 1, n)).astype(int)]


ref = qpos36_feature_to_omg(build_qpos36_from_artifact(f"{ART}/{CROUCH}/motion.npz"))
n = len(ref)
bfm_orig = resample(np.load(f"{HERE}/quant/bfm_0.npz")["executed_qpos_36"], n)
bfm_dec = resample(np.load(f"{HERE}/quant/dec_0.npz")["executed_qpos_36"], n)

panels = []
for tag, q, title in [("ref", ref, "Reference"),
                      ("orig", bfm_orig, "BFM-Zero: original"),
                      ("dec", bfm_dec, "BFM-Zero: VAE-decoded")]:
    p = f"{HERE}/_pc_{tag}.mp4"
    render_qpos_video(q[:n], p, fps=30, width=640, height=720, title=title)
    panels.append(p)

out = f"{HERE}/pipeline_crouch.mp4"
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", panels[0], "-i", panels[1], "-i", panels[2],
                "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out], check=True)
for p in panels:
    os.remove(p)
print(f"wrote {out} ({n} frames)")
