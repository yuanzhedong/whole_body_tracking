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
    T = len(data); eul = np.zeros((T, len(names), 3)); root_pos = np.zeros((T, 3)); col = 0
    for j, c in enumerate(chans):
        if c == 6:
            if j == 0:
                root_pos = data[:, col:col+3]
            eul[:, j] = data[:, col+3:col+6]
        elif c == 3:
            eul[:, j] = data[:, col:col+3]
        col += c
    return eul, root_pos


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

    eul_full, rootp_full = parse_bvh(a.bvh)
    idx = np.round(np.linspace(0, len(eul_full) - 1, a.n_frames)).astype(int)
    eul = eul_full[idx]; transl_np = (rootp_full[idx] / 100.0).astype(np.float32)   # cm -> m, root translation
    T = len(eul)
    Rloc = R.from_euler("ZYX", eul.reshape(-1, 3), degrees=True).as_matrix().reshape(T, -1, 3, 3)
    verts = []
    for s in range(0, T, 64):
        local = torch.tensor(Rloc[s:s+64], dtype=torch.float32, device=dev)
        world = joint_local_to_world(local, parents) @ correction
        localc = joint_world_to_local(world, parents)
        pose = torch.cat([localc[:, 1:2], localc[:, 2:]], dim=1)     # [B,77,3,3]
        tb = torch.tensor(transl_np[s:s+64], device=dev)
        out = sl.pose(pose, transl=tb, pose2rot=False)
        verts.append(out["vertices"].detach().cpu().numpy())
    verts = np.concatenate(verts, 0)

    # checkerboard ground (fixed in world) so the translation is visible under a tracking camera
    def ground(y, xmin, xmax, zmin, zmax, tile=0.5):
        import trimesh as _tm
        tiles = []
        xs = np.arange(np.floor(xmin/tile)*tile, xmax + tile, tile)
        zs = np.arange(np.floor(zmin/tile)*tile, zmax + tile, tile)
        for ix, x in enumerate(xs):
            for iz, z in enumerate(zs):
                q = _tm.Trimesh(vertices=[[x, y, z], [x+tile, y, z], [x+tile, y, z+tile], [x, y, z+tile]],
                                faces=[[0, 2, 1], [0, 3, 2]], process=False)   # normals up (+y)
                c = [175, 185, 200, 255] if (ix + iz) % 2 else [120, 135, 160, 255]
                q.visual.vertex_colors = c; tiles.append(q)
        return _tm.util.concatenate(tiles)

    feet_y = float(verts[:, :, 1].min())
    gx0, gx1 = verts[:, :, 0].min() - 1.5, verts[:, :, 0].max() + 1.5
    gz0, gz1 = verts[:, :, 2].min() - 1.5, verts[:, :, 2].max() + 1.5
    gmesh = pyrender.Mesh.from_trimesh(ground(feet_y, gx0, gx1, gz0, gz1), smooth=False)

    w = h = a.size
    r = pyrender.OffscreenRenderer(w, h); frames = []
    for t in range(T):
        V = verts[t]
        mesh = trimesh.Trimesh(V, faces, process=False)
        mesh.visual.vertex_colors = [230, 100, 200, 255]
        scene = pyrender.Scene(bg_color=[0.05, 0.07, 0.12, 1], ambient_light=[.55, .55, .55])
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
        scene.add(gmesh)
        ctr = V.mean(0)
        cam = np.eye(4)
        Rc = R.from_euler("x", -18, degrees=True).as_matrix()   # slight downward tilt (see the ground)
        cam[:3, :3] = Rc; cam[:3, 3] = [ctr[0], ctr[1] + 0.9, ctr[2] + 2.9]
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi/3), pose=cam)
        scene.add(pyrender.DirectionalLight(intensity=4), pose=cam)
        col, _ = r.render(scene); frames.append(col.copy())
    r.delete()
    np.save(a.out_npy, np.stack(frames))
    print(f"human mesh frames -> {a.out_npy} {np.stack(frames).shape}")


if __name__ == "__main__":
    main()
