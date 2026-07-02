"""Standalone human-skeleton renderer (pyrender only — no mujoco, to avoid EGL clash).

BVH -> FK joint positions -> skeleton frames saved as .npy [T,H,W,3]. Called as a
subprocess by render_triptych.py. Runs in .venv6.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
try:                       # initializing warp sets up the CUDA/EGL device pyrender needs
    import warp as _wp; _wp.init()
except Exception:
    pass
import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_bvh_full(path):
    text = Path(path).read_text().splitlines()
    names, chans, parents, offsets = [], [], [], []
    stack, cur = [], -1
    i = 0
    while "MOTION" not in text[i]:
        s = text[i].strip()
        if s.startswith(("ROOT", "JOINT")):
            parents.append(stack[-1] if stack else -1); names.append(s.split()[1])
            chans.append(0); offsets.append((0, 0, 0)); cur = len(names) - 1
        elif s.startswith("OFFSET") and cur >= 0 and len(offsets) == len(names):
            offsets[cur] = tuple(float(x) for x in s.split()[1:4])
        elif s == "{": stack.append(cur)
        elif s == "}": stack.pop()
        elif s.startswith("CHANNELS"): chans[-1] = int(s.split()[1])
        elif s.startswith("End"): cur = -2
        i += 1
    while "Frame Time" not in text[i]: i += 1
    data = np.array([[float(x) for x in ln.split()] for ln in text[i+1:] if ln.strip()])
    T = len(data); root_pos = np.zeros((T, 3)); eul = np.zeros((T, len(names), 3)); col = 0
    for j, c in enumerate(chans):
        if c == 6:
            if j == 0: root_pos = data[:, col:col+3]
            eul[:, j] = data[:, col+3:col+6]
        elif c == 3: eul[:, j] = data[:, col:col+3]
        col += c
    return parents, np.array(offsets), root_pos, eul


def fk_positions(parents, offsets, root_pos, eul):
    T, J = eul.shape[0], len(parents)
    Rl = R.from_euler("ZYX", eul.reshape(-1, 3), degrees=True).as_matrix().reshape(T, J, 3, 3)
    pos = np.zeros((T, J, 3)); Rabs = np.zeros((T, J, 3, 3))
    for j in range(J):
        p = parents[j]
        if p < 0:
            Rabs[:, j] = Rl[:, j]; pos[:, j] = root_pos
        else:
            Rabs[:, j] = np.einsum("tab,tbc->tac", Rabs[:, p], Rl[:, j])
            pos[:, j] = pos[:, p] + np.einsum("tab,b->ta", Rabs[:, p], offsets[j])
    return pos / 100.0


def resample(a, n):
    return a[np.round(np.linspace(0, len(a) - 1, n)).astype(int)]


def main():
    import trimesh, pyrender
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", required=True); p.add_argument("--out-npy", required=True)
    p.add_argument("--n-frames", type=int, required=True); p.add_argument("--size", type=int, default=480)
    a = p.parse_args()
    parents, offsets, root_pos, eul = parse_bvh_full(a.bvh)
    pos = fk_positions(parents, offsets, resample(root_pos, a.n_frames), resample(eul, a.n_frames))
    w = h = a.size
    r = pyrender.OffscreenRenderer(w, h); frames = []
    for t in range(len(pos)):
        P = pos[t]; meshes = []
        for j, par in enumerate(parents):
            sph = trimesh.creation.uv_sphere(radius=0.02); sph.apply_translation(P[j])
            sph.visual.vertex_colors = [230, 100, 200, 255]; meshes.append(sph)
            if par >= 0:
                aa, bb = P[par], P[j]; L = np.linalg.norm(bb - aa)
                if L > 1e-4:
                    cyl = trimesh.creation.cylinder(radius=0.013, height=L)
                    d = (bb - aa) / L; z = np.array([0, 0, 1.0]); v = np.cross(z, d); s = np.linalg.norm(v)
                    Rm = np.eye(3) if s < 1e-6 else R.from_rotvec(v/s*np.arccos(np.clip(z@d, -1, 1))).as_matrix()
                    Tm = np.eye(4); Tm[:3, :3] = Rm; Tm[:3, 3] = (aa+bb)/2; cyl.apply_transform(Tm)
                    cyl.visual.vertex_colors = [205, 85, 175, 255]; meshes.append(cyl)
        scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[.6, .6, .6])
        scene.add(pyrender.Mesh.from_trimesh(trimesh.util.concatenate(meshes), smooth=False))
        ctr = P.mean(0); cam = np.eye(4); cam[:3, 3] = [ctr[0], ctr[1], ctr[2] + 3.0]
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi/3), pose=cam)
        scene.add(pyrender.DirectionalLight(intensity=4), pose=cam)
        col, _ = r.render(scene); frames.append(col.copy())
    r.delete()
    np.save(a.out_npy, np.stack(frames))
    print(f"human frames -> {a.out_npy} {np.stack(frames).shape}")


if __name__ == "__main__":
    main()
