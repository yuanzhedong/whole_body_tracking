"""Render the main pipeline diagram: BONES-SEED -> UniMoTok VAE -> HoloMotion sim2sim.

Stage-1 of the omni-modal effort: learn a G1 motion latent (UniMoTok VAE) and
validate that decoded motion is physically executable via the HoloMotion tracker in
MuJoCo. Writes pipeline_main.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "#4a78b5"    # data / representation (blue)
VAE = "#7a4fb5"     # the learned latent (purple — the contribution)
VAL = "#d98a3c"     # physics validator (orange — reused from OMG)
GREY = "#555555"


def box(ax, x, y, w, h, text, ec, fc="white", fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 linewidth=2, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", color="#111111")


def arrow(ax, x0, y0, x1, y1, color=GREY, label=None, lx=0, ly=0.18):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=16, linewidth=2, color=color, zorder=1))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, ha="center",
                va="center", fontsize=8.5, style="italic", color=GREY)


fig, ax = plt.subplots(figsize=(13.5, 7.8))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.8); ax.axis("off")

ax.text(6.0, 7.5, "Stage 1 — learn a G1 motion latent, then validate it is physically executable",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color="#222222")

# Row A (top): dataset -> features -> VAE latent
box(ax, 2.1, 5.5, 3.4, 1.05, "BONES-SEED\nG1 motion dataset\n(~142k clips, 30 Hz)", DATA, fc="#eaf1fb")
arrow(ax, 3.85, 5.5, 4.95, 5.5, label="export")
box(ax, 6.75, 5.5, 3.2, 1.05, "41-D motion features\nroot rot6d + root vel\n+ 29 joint angles", DATA, fc="#eaf1fb", fs=9.8)
arrow(ax, 8.4, 5.5, 9.55, 5.5, label="encode")
box(ax, 11.3, 5.5, 3.4, 1.15, "UniMoTok VAE latent  z\n(512-d, continuous)\n— the motion tokenizer —",
    VAE, fc="#f1ebf8", bold=True, fs=10)

# wrap down from VAE to Row B
arrow(ax, 11.3, 4.92, 11.3, 4.35, VAE, label="decode", lx=0.95, ly=0)

# Row B (bottom): qpos -> HoloMotion -> MuJoCo -> metrics
box(ax, 11.3, 3.75, 3.4, 1.05, "decode → qpos_36\nroot pose + 29 dof\n(reference motion)", DATA, fc="#eaf1fb", fs=9.8)
arrow(ax, 9.6, 3.75, 8.5, 3.75, label="reference")
box(ax, 6.75, 3.75, 3.2, 1.05, "HoloMotion tracker\ngeneralist G1 policy\n(ONNX, 30→50 Hz)", VAL, fc="#fbf0e3", fs=9.8)
arrow(ax, 5.15, 3.75, 4.05, 3.75, label="actions")
box(ax, 2.3, 3.75, 3.2, 1.05, "MuJoCo G1\nphysics rollout", VAL, fc="#fbf0e3", bold=True)
arrow(ax, 2.3, 3.22, 2.3, 2.62, label="executed", lx=1.0, ly=0)
box(ax, 2.3, 2.15, 3.7, 0.95, "rollout_metrics\nsurvival + tracking error", GREY, fc="white", fs=9.8)

# omni-model note hanging off the VAE latent
ax.add_patch(FancyArrowPatch((11.3, 6.08), (11.3, 6.62), arrowstyle="-|>",
             mutation_scale=14, linewidth=1.6, linestyle=(0, (4, 3)), color=VAE, zorder=1))
ax.text(11.3, 6.85, "↑ interface for future\ntext / multimodal → motion (omni-model)",
        ha="center", va="center", fontsize=8.6, style="italic", color=VAE)

# legend
ax.text(0.2, 0.7, "■", color=DATA, fontsize=13); ax.text(0.55, 0.7, "data / representation", fontsize=9, va="center")
ax.text(4.0, 0.7, "■", color=VAE, fontsize=13); ax.text(4.35, 0.7, "learned latent (this work)", fontsize=9, va="center")
ax.text(8.2, 0.7, "■", color=VAL, fontsize=13); ax.text(8.55, 0.7, "physics validator (HoloMotion + MuJoCo)", fontsize=9, va="center")

fig.tight_layout()
out = f"{HERE}/pipeline_main.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
