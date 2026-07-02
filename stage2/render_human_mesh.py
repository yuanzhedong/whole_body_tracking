"""Standalone SOMA/MHR human MESH renderer from a BONES-SEED soma_uniform BVH -> frames .npy.

Uses the demo_soma_vis.py convention: BVH local rotations -> FK to world (SOMA parent tree)
-> apply rest-frame correction (inverse of SOMA T-pose world rotation) -> back to local ->
model.pose(pose2rot=False). Bundled MHR model, no gated SMPL. Called (subprocess) by
render_triptych.py; run in .venv6.
"""
import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
try:
    import warp as _wp; _wp.init()          # sets up the EGL device pyrender needs
except Exception:
    pass
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

ASSETS = "/ws/user/yzdong/src/github/SOMA-X/assets"
SHAPES = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/soma_shapes/soma_proportion_fit_mhr_params"


def parse_bvh(path):
    text = Path(path).read_text().splitlines()
    names, chans = [], []
    i = 0
    while i < len(text) and "MOTION" not in text[i]:
        s = text[i].strip()
        if s.startswith(("ROOT", "JOINT")):
            names.append(s.split()[1]); chans.append(0)
        elif s.startswith("CHANNELS"):
            chans[-1] = int(s.split()[1])
        i += 1
    while "Frame Time" not in text[i]:
        i += 1
    data = np.array([[float(x) for x in ln.split()] for ln in text[i+1:] if ln.strip()])
    T = len(data); eul = np.zeros((T, len(names), 3)); col = 0
    for j, c in enumerate(chans):
        if c == 6:
            eul[:, j] = data[:, col+3:col+6]
        elif c == 3:
            eul[:, j] = data[:, col:col+3]
        col += c
    return eul


def resample(a, n):
    return a[np.round(np.linspace(0, len(a) - 1, n)).astype(int)]


def main():
    from soma import SomaLayer
    from soma.geometry.rig_utils import joint_local_to_world, joint_world_to_local
    import trimesh, pyrender

    p = argparse.ArgumentParser()
    p.add_argument("--bvh", required=True); p.add_argument("--out-npy", required=True)
    p.add_argument("--n-frames", type=int, required=True); p.add_argument("--size", type=int, default=480)
    a = p.parse_args()
    dev = "cuda"

    actor = re.search(r"(A\d+)", Path(a.bvh).stem)
    sp = Path(SHAPES) / f"{actor.group(1) if actor else 'A057'}.npz"
    if not sp.exists():
        sp = Path(SHAPES).parent / "soma_base_fit_mhr_params.npz"
    d = np.load(sp)
    sl = SomaLayer(data_root=ASSETS, identity_model_type="mhr", low_lod=True, device=dev)
    idc = torch.tensor(d["identity_params"], dtype=torch.float32, device=dev)
    scl = torch.tensor(d["scale_params"], dtype=torch.float32, device=dev)
    faces = np.asarray(sl.faces.detach().cpu() if hasattr(sl.faces, "detach") else sl.faces)
    parents = sl.public_joint_parent_ids
    rest_world = sl.t_pose_world[sl.public_transform_joint_indices]
    correction = rest_world[:, :3, :3].transpose(-2, -1)
    sl.prepare_identity(idc, scl, repose_to_bind_pose=True)

    eul = resample(parse_bvh(a.bvh), a.n_frames)          # [T,78,3] ZYX deg (local)
    T = len(eul)
    Rloc = R.from_euler("ZYX", eul.reshape(-1, 3), degrees=True).as_matrix().reshape(T, -1, 3, 3)
    verts = []
    for s in range(0, T, 64):
        local = torch.tensor(Rloc[s:s+64], dtype=torch.float32, device=dev)
        world = joint_local_to_world(local, parents) @ correction
        localc = joint_world_to_local(world, parents)
        pose = torch.cat([localc[:, 1:2], localc[:, 2:]], dim=1)     # [B,77,3,3]
        out = sl.pose(pose, transl=torch.zeros(pose.shape[0], 3, device=dev), pose2rot=False)
        verts.append(out["vertices"].detach().cpu().numpy())
    verts = np.concatenate(verts, 0)

    w = h = a.size
    r = pyrender.OffscreenRenderer(w, h); frames = []
    for t in range(T):
        V = verts[t]
        mesh = trimesh.Trimesh(V, faces, process=False)
        mesh.visual.vertex_colors = [230, 100, 200, 255]
        scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[.5, .5, .5])
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
        ctr = V.mean(0); cam = np.eye(4); cam[:3, 3] = [ctr[0], ctr[1], ctr[2] + 2.6]
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi/3), pose=cam)
        scene.add(pyrender.DirectionalLight(intensity=4), pose=cam)
        col, _ = r.render(scene); frames.append(col.copy())
    r.delete()
    np.save(a.out_npy, np.stack(frames))
    print(f"human mesh frames -> {a.out_npy} {np.stack(frames).shape}")


if __name__ == "__main__":
    main()
