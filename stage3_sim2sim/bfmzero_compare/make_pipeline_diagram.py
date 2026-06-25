"""Render the tracker-comparison pipeline diagram (each tracker's input format).

Both trackers receive the SAME reference G1 motion; the diagram shows how that one
reference enters each tracker in its native input format, then the shared physics
and scoring. Writes pipeline.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
HOLO = "#d9534f"   # red-ish: fails near-ground
BFM = "#3c9a5f"    # green-ish: succeeds
NEU = "#4a78b5"    # shared/neutral
GREY = "#555555"


def box(ax, x, y, w, h, text, ec, fc="white", fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=2, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", color="#111111")


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=18, linewidth=2, color=color, zorder=1))


fig, ax = plt.subplots(figsize=(13, 8.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 8.5); ax.axis("off")

# shared reference
box(ax, 6.5, 7.8, 8.4, 1.0,
    "BONES-SEED G1 reference clip\nqpos_36  =  [ root pos (3) + quat (4)  |  29 joint angles ]",
    NEU, fc="#eaf1fb", fs=12, bold=True)
ax.text(6.5, 7.05, "same reference motion fed to both trackers (two input formats)",
        ha="center", va="center", fontsize=9.5, style="italic", color=GREY)

# split arrows
arrow(ax, 5.5, 7.25, 3.2, 6.35, HOLO)
arrow(ax, 7.5, 7.25, 9.8, 6.35, BFM)

# HoloMotion column
box(ax, 3.2, 5.7, 5.0, 1.35,
    "HoloMotion input\n522-d tracking observation:\nreference joint targets + root,\n"
    "robot proprioception + history\n(FEATURE joint order)", HOLO, fc="#fbeae9", fs=9.5)
arrow(ax, 3.2, 4.97, 3.2, 4.35, HOLO)
box(ax, 3.2, 3.85, 4.4, 0.9, "HoloMotion\ngeneralist tracker (ONNX, v1.3.1)", HOLO, bold=True, fs=11)

# BFM-Zero column
box(ax, 9.8, 5.7, 5.2, 1.35,
    "BFM-Zero input\ndof → pose_aa (robot axis-angle),\nrobot-FK reference,\n"
    "FB backward-map → z latent\n(OMG joint order)", BFM, fc="#e9f4ee", fs=9.5)
arrow(ax, 9.8, 4.97, 9.8, 4.35, BFM)
box(ax, 9.8, 3.85, 4.6, 0.9, "BFM-Zero\nForward-Backward foundation model", BFM, bold=True, fs=11)

# merge to shared physics
arrow(ax, 3.2, 3.38, 5.7, 2.65, GREY)
arrow(ax, 9.8, 3.38, 7.3, 2.65, GREY)
box(ax, 6.5, 2.25, 5.2, 0.85, "MuJoCo G1 physics  (identical for both)", NEU, fc="#eaf1fb", bold=True, fs=11)
arrow(ax, 6.5, 1.8, 6.5, 1.25, GREY)
box(ax, 6.5, 0.78, 6.8, 0.85,
    "executed qpos_36  →  rollout_metrics  (survival, ref-relative survival, joint error)",
    GREY, fc="white", fs=10)

fig.tight_layout()
out = f"{HERE}/pipeline.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
