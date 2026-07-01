"""Self-contained MuJoCo renderer for G1 motion -> side-by-side dataset videos.

Renders qpos_36 = [root_pos(3), root_quat_wxyz(4), joints(29)] (MuJoCo/OMG joint order)
with a root-tracking camera, and stitches panels horizontally (e.g. Reference | BFM-Zero
executed) into an mp4 -- the robot side of the DATA_PIPELINE.md sample videos.

Input is a batch_tracking_inference rollout npz (executed_qpos_36 + reference_qpos_36,
both already in MuJoCo order). Runs anywhere with mujoco + imageio (e.g. .venv-bfm).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

XML = ("/ws/user/yzdong/src/github/BFM-Zero/humanoidverse/data/robots/g1/"
       "scene_29dof_freebase_noadditional_actuators.xml")


def render_qpos(qpos, xml=XML, w=480, h=480, fps=30, azimuth=135.0, elevation=-15.0, distance=3.0):
    """qpos[T,36] -> frames[T,h,w,3] with a camera that follows the root."""
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    nq = model.nq
    frames = []
    for t in range(len(qpos)):
        q = np.asarray(qpos[t], np.float64)
        data.qpos[:] = q[:nq] if len(q) >= nq else np.pad(q, (0, nq - len(q)))
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[:3]              # track the pelvis
        cam.lookat[2] = max(cam.lookat[2], 0.6)
        renderer.update_scene(data, cam)
        frames.append(renderer.render().copy())
    renderer.close()
    return np.stack(frames)


def label(frames, text):
    if not (HAVE_CV2 and text):
        return frames
    out = frames.copy()
    for f in out:
        cv2.putText(f, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", required=True, help="batch_tracking_inference rollout npz")
    p.add_argument("--out", required=True, help="output mp4")
    p.add_argument("--fps", type=int, default=30, help="output fps (real-time)")
    p.add_argument("--src-fps", type=int, default=50, help="rollout data rate (BFM-Zero = 50 Hz)")
    p.add_argument("--panels", default="ref,exec", help="comma of {ref,exec}; order = left->right")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = p.parse_args()

    d = np.load(args.rollout)
    ex = np.asarray(d["executed_qpos_36"], np.float32)
    ref = np.asarray(d["reference_qpos_36"], np.float32)
    n = min(len(ex), len(ref))
    ex, ref = ex[:n], ref[:n]
    if args.src_fps != args.fps:                       # resample to real-time output fps
        m = max(2, int(round(n * args.fps / args.src_fps)))
        idx = np.round(np.linspace(0, n - 1, m)).astype(int)
        ex, ref = ex[idx], ref[idx]
    if args.max_frames:
        ex, ref = ex[:args.max_frames], ref[:args.max_frames]

    src = {"ref": (ref, "Reference (retarget)"), "exec": (ex, "BFM-Zero executed")}
    panels = [src[k.strip()] for k in args.panels.split(",") if k.strip() in src]
    rendered = [label(render_qpos(q, fps=args.fps), title) for q, title in panels]
    combined = np.concatenate(rendered, axis=2)  # hstack panels
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, combined, fps=args.fps, quality=8, macro_block_size=1)
    print(f"wrote {args.out}  ({combined.shape[0]} frames, {combined.shape[2]}x{combined.shape[1]})")


if __name__ == "__main__":
    main()
