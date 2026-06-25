"""Per-clip triptych: ONE reference + both trackers' executions, side by side.

Layout: [ Reference | HoloMotion | BFM-Zero ] in a single row (no duplicated
reference). All three single-panel renders share the OMG MuJoCo renderer; the
reference is the ground-truth artifact qpos (rendered once). Run in the OMG env.
"""
import glob
import os
import subprocess
import numpy as np

from stage3_sim2sim.sim2sim import build_qpos36_from_artifact
from stage3_sim2sim.joint_order import qpos36_feature_to_omg
from omg.render.mujoco import render_qpos_video

HERE = os.path.dirname(os.path.abspath(__file__))
ART = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/artifacts_seed_full"
FFMPEG = "/usr/bin/ffmpeg"
CLIPS = [
    # only the two replaced (grounded) clips; clip2/clip3 triptychs are unchanged
    ("clip0", "crouch_ff_start_180_R_003__A145_M:v0"),
    ("clip1", "squat_002__A359:v0"),
]


def resample(a, n):
    return a[np.round(np.linspace(0, len(a) - 1, n)).astype(int)]


def holo_rollout(cid):
    return glob.glob(f"/tmp/holo_{cid}/**/holomotion_rollout.npz", recursive=True)[0]


for cid, art in CLIPS:
    ref = qpos36_feature_to_omg(build_qpos36_from_artifact(f"{ART}/{art}/motion.npz"))
    n = len(ref)
    holo_ex = np.load(holo_rollout(cid))["executed_qpos_36"][:n]
    bfm_ex = resample(np.load(f"{HERE}/rollout_{cid}.npz")["executed_qpos_36"], n)

    panels = []
    for tag, q, title in [("ref", ref, "Reference"), ("holo", holo_ex, "HoloMotion"),
                          ("bfm", bfm_ex, "BFM-Zero")]:
        p = f"{HERE}/_panel_{cid}_{tag}.mp4"
        render_qpos_video(q[:n], p, fps=30, width=640, height=720, title=title)
        panels.append(p)

    out = f"{HERE}/triptych_{cid}.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", panels[0], "-i", panels[1],
                    "-i", panels[2], "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]",
                    "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out],
                   check=True)
    for p in panels:
        os.remove(p)
    print(f"{cid}: {out}  ({n} frames)")
print("done")
