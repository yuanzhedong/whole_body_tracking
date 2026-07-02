"""Build the DATA_PIPELINE triptych: [ human motion | G1 reference | G1 executed ].

Pure orchestrator (mujoco EGL and pyrender EGL can't share one env):
  * robot panels  -> robot_frames.py in .venv-bfm  (working mujoco EGL)
  * human panel   -> render_human_mesh.py in .venv6 (SOMA/MHR mesh, pyrender)
then stitches the frames into an mp4. Run with any python.
"""
import argparse
import glob
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

HERE = Path(__file__).resolve().parent
BVH_ROOT = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/soma_uniform/bvh"
VENV6 = "/ws/user/yzdong/src/github/whole_body_tracking/.venv6/bin/python"
VENVBFM = "/ws/user/yzdong/src/github/BFM-Zero/.venv-bfm/bin/python"


def label(frames, text, color=(0, 0, 0)):
    if not (HAVE_CV2 and text):
        return frames
    out = frames.copy()
    for f in out:
        cv2.putText(f, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--rollout", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--gpu", default="2")
    args = p.parse_args()

    bvh = glob.glob(f"{BVH_ROOT}/*/{args.clip}.bvh")
    if not bvh:
        raise FileNotFoundError(f"no BVH for {args.clip}")
    d = np.load(args.rollout)
    n = min(len(d["executed_qpos_36"]), len(d["reference_qpos_36"]))
    n_out = max(2, int(round(n * args.fps / 50)))     # robot 50Hz -> real-time
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=args.gpu)

    hum_npy = tempfile.mktemp(suffix=".npy")
    rob_npz = tempfile.mktemp(suffix=".npz")
    subprocess.run([VENV6, str(HERE / "render_human_mesh.py"), "--bvh", bvh[0],
                    "--out-npy", hum_npy, "--n-frames", str(n_out)],
                   check=True, env=dict(env, PYOPENGL_PLATFORM="egl"))
    subprocess.run([VENVBFM, str(HERE / "robot_frames.py"), "--rollout", args.rollout,
                    "--out-npz", rob_npz, "--n-frames", str(n_out)],
                   check=True, env=dict(env, MUJOCO_GL="egl", PYTHONPATH="."))

    hum = label(np.load(hum_npy), "Human (BONES-SEED)", color=(255, 255, 255))
    rob = np.load(rob_npz)
    ref_f = label(rob["ref"], "G1 reference")
    ex_f = label(rob["exec"], "BFM-Zero executed")
    os.remove(hum_npy); os.remove(rob_npz)

    m = min(len(hum), len(ref_f), len(ex_f))
    trip = np.concatenate([hum[:m], ref_f[:m], ex_f[:m]], axis=2)
    import imageio.v2 as imageio
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, trip, fps=args.fps, quality=8, macro_block_size=1)
    print(f"wrote {args.out} ({trip.shape[0]} frames, {trip.shape[2]}x{trip.shape[1]})")


if __name__ == "__main__":
    main()
