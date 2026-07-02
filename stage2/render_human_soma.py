"""Render the SOMA/MHR human mesh from a BONES-SEED soma_uniform BVH -> frames/mp4.

BVH joints == SOMA public_joint_names (exact order, 78). poses = joints[1:] (77) as
axis-angle; Root gives global translation + orientation (folded into Hips). Uses SOMA-X's
bundled MHR model (low-lod) + the actor's MHR shape params (soma_shapes). This is the
left ("human") panel of the DATA_PIPELINE triptych — no gated SMPL models needed.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from scipy.spatial.transform import Rotation as R

ASSETS = "/ws/user/yzdong/src/github/SOMA-X/assets"
SHAPES = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/soma_shapes/soma_proportion_fit_mhr_params"


def parse_bvh(path):
    """Return names, parents, root_pos[T,3], euler[T,J,3] (ZYX deg), fps. Parents from the tree."""
    text = Path(path).read_text().splitlines()
    names, chans, parents = [], [], []
    stack = []
    i = 0
    while i < len(text) and "MOTION" not in text[i]:
        line = text[i].strip()
        if line.startswith(("ROOT", "JOINT")):
            parents.append(stack[-1] if stack else -1)
            names.append(line.split()[1]); chans.append(0)
            cur = len(names) - 1
        elif line == "{":
            stack.append(cur)
        elif line == "}":
            stack.pop()
        elif line.startswith("CHANNELS"):
            chans[-1] = int(line.split()[1])
        elif line.startswith("End"):        # End Site (no channels, skip its braces)
            cur = -2
        i += 1
    # MOTION
    while "Frame Time" not in text[i]:
        i += 1
    fps = round(1.0 / float(text[i].split(":")[1]))
    data = np.array([[float(x) for x in ln.split()] for ln in text[i + 1:] if ln.strip()])
    T = data.shape[0]
    # walk channel columns
    root_pos = np.zeros((T, 3)); eul = np.zeros((T, len(names), 3))
    col = 0
    for j, c in enumerate(chans):
        if c == 6:
            if j == 0:
                root_pos = data[:, col:col + 3]
            eul[:, j] = data[:, col + 3:col + 6]      # Zrot,Yrot,Xrot
        elif c == 3:
            eul[:, j] = data[:, col:col + 3]
        col += c
    return names, parents, root_pos, eul, fps


def bvh_to_soma(parents, root_pos, eul):
    """FK BVH local rotations -> ABSOLUTE world rotations; return poses[T,77,3] axis-angle
    (joints[1:], absolute) + transl[T,3] (m), for SOMA forward(absolute_pose=True)."""
    T, J, _ = eul.shape
    Rloc = R.from_euler("ZYX", eul.reshape(-1, 3), degrees=True).as_matrix().reshape(T, J, 3, 3)
    Rabs = np.zeros_like(Rloc)
    for j in range(J):                       # joints are in topological (parent-before-child) order
        p = parents[j]
        Rabs[:, j] = Rloc[:, j] if p < 0 else np.einsum("tij,tjk->tik", Rabs[:, p], Rloc[:, j])
    poses = R.from_matrix(Rabs[:, 1:].reshape(-1, 3, 3)).as_rotvec().reshape(T, J - 1, 3).astype(np.float32)
    transl = (root_pos / 100.0).astype(np.float32)     # BVH cm -> m
    return poses, transl


def render_frames(verts, faces, w=480, h=480, color=(230, 100, 200)):
    import trimesh, pyrender
    faces = np.asarray(faces)
    frames = []
    scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[.45, .45, .45])
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3)
    cam_node = scene.add(cam, pose=np.eye(4))
    light_node = scene.add(pyrender.DirectionalLight(intensity=4.0), pose=np.eye(4))
    r = pyrender.OffscreenRenderer(w, h)
    mnode = None
    for t in range(len(verts)):
        V = verts[t]
        mesh = trimesh.Trimesh(V, faces, process=False)
        mesh.visual.vertex_colors = list(color) + [255]
        if mnode is not None:
            scene.remove_node(mnode)
        mnode = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
        ctr = V.mean(0)
        pose = np.eye(4); pose[:3, 3] = [ctr[0], ctr[1], ctr[2] + 3.2]
        scene.set_pose(cam_node, pose); scene.set_pose(light_node, pose)
        col, _ = r.render(scene)
        frames.append(col.copy())
    r.delete()
    return np.stack(frames)


def main():
    import torch
    from soma import SomaLayer
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=4, help="120fps BVH -> 30fps default")
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    actor = re.search(r"(A\d+)", Path(args.bvh).stem)
    actor = actor.group(1) if actor else "A057"
    sp = Path(SHAPES) / f"{actor}.npz"
    if not sp.exists():
        sp = Path(SHAPES).parent / "soma_base_fit_mhr_params.npz"
    d = np.load(sp)
    names, parents, root_pos, eul, _ = parse_bvh(args.bvh)
    root_pos, eul = root_pos[::args.stride], eul[::args.stride]
    if args.max_frames:
        root_pos, eul = root_pos[:args.max_frames], eul[:args.max_frames]
    poses, transl = bvh_to_soma(parents, root_pos, eul)

    sl = SomaLayer(data_root=ASSETS, identity_model_type="mhr", low_lod=True, device="cuda")
    idc = torch.tensor(d["identity_params"], dtype=torch.float32, device="cuda")
    scl = torch.tensor(d["scale_params"], dtype=torch.float32, device="cuda")
    faces = sl.faces.detach().cpu().numpy() if hasattr(sl.faces, "detach") else np.asarray(sl.faces)

    verts = []
    for s in range(0, len(poses), 64):
        pb = torch.tensor(poses[s:s + 64], device="cuda")
        tb = torch.tensor(transl[s:s + 64], device="cuda")
        B = pb.shape[0]
        out = sl.forward(pb, idc.repeat(B, 1), scale_params=scl.repeat(B, 1), transl=tb,
                         pose2rot=True, absolute_pose=True)
        verts.append(out["vertices"].detach().cpu().numpy())
    verts = np.concatenate(verts, 0)
    frames = render_frames(verts, faces)
    import imageio.v2 as imageio
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=args.fps, quality=8, macro_block_size=1)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
