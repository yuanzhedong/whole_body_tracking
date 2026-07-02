"""Render G1 reference + BFM-Zero-executed frames from a rollout npz -> npz of frames.

Runs in .venv-bfm (working mujoco EGL). Used by render_triptych.py (which orchestrates the
human panel separately in .venv6, since the two EGL backends don't coexist in one env).
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_g1_clip import render_qpos


def resample(a, n):
    return a[np.round(np.linspace(0, len(a) - 1, n)).astype(int)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--n-frames", type=int, required=True)
    a = p.parse_args()
    d = np.load(a.rollout)
    ex = resample(np.asarray(d["executed_qpos_36"], np.float32), a.n_frames)
    ref = resample(np.asarray(d["reference_qpos_36"], np.float32), a.n_frames)
    np.savez(a.out_npz, ref=render_qpos(ref), exec=render_qpos(ex))
    print(f"robot frames -> {a.out_npz}")


if __name__ == "__main__":
    main()
