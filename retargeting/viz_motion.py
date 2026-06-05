"""Render a motion npz (from scripts/csv_to_npz.py) to MP4 by reusing the repo's decoupled
Isaac-Sim-6.0 renderer (tools/render_rollout_sim6.py). Interactive Isaac rendering segfaults on
this box's driver, so we go through the same kinematic-replay path used for policy rollouts.

The motion npz stores body_pos_w/body_quat_w (wxyz) for all 30 G1 bodies but no body names or
root pose, so this script repackages it into the states-npz schema the renderer expects
(root_pos, body_pos, body_quat, body_names) and shells out to the renderer + ffmpeg.

Usage (pin a 4090 — Blackwell has no kernels in this stack):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
    python retargeting/viz_motion.py --npz /tmp/motion.npz --out retargeting/out/amass_walk.mp4
"""
import argparse
import os
import subprocess
import sys
import numpy as np

REPO = "/ws/user/yzdong/src/github/whole_body_tracking"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_bodies import G1_BODY_NAMES, PELVIS_IDX  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="motion npz from csv_to_npz.py")
    p.add_argument("--out", default="retargeting/out/motion.mp4")
    p.add_argument("--camera", default="treadmill", choices=["treadmill", "follow"])
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--workdir", default="/tmp/wbt_amass_viz")
    args = p.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    frames_dir = os.path.join(args.workdir, "frames")
    states = os.path.join(args.workdir, "states.npz")
    result = os.path.join(args.workdir, "render.txt")

    d = np.load(args.npz)
    bp = d["body_pos_w"].astype(np.float64)     # [T, 30, 3]
    bq = d["body_quat_w"].astype(np.float64)    # [T, 30, 4] wxyz
    assert bp.shape[1] == len(G1_BODY_NAMES), f"expected {len(G1_BODY_NAMES)} bodies, got {bp.shape[1]}"
    np.savez(
        states,
        root_pos=bp[:, PELVIS_IDX, :],          # for treadmill xy re-centering
        body_pos=bp, body_quat=bq,
        body_names=np.array(G1_BODY_NAMES),
    )

    # render in Isaac Sim 6.0 (.venv6) via the existing renderer, then encode
    venv6 = os.path.join(REPO, ".venv6/bin/python")
    renderer = os.path.join(REPO, "tools/render_rollout_sim6.py")
    subprocess.run(
        [venv6, renderer, "--states", states, "--out_dir", frames_dir,
         "--usd_dir", os.path.join(args.workdir, "g1_usd"), "--camera", args.camera,
         "--result", result, "--res", "1280", "720", "--stride", str(args.stride)],
        check=True, cwd=REPO,
    )
    if "RENDER_OK" not in open(result).read():
        print("RENDER FAILED:\n" + open(result).read()); sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(args.fps), "-pattern_type", "glob",
         "-i", os.path.join(frames_dir, "rgb_*.png"), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
        check=True, stderr=subprocess.DEVNULL,
    )
    print(f"DONE -> {args.out}")


if __name__ == "__main__":
    main()
