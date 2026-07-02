"""L0 keystone: synthetic qpos -> build_features -> invert -> recover (Gate A).

This is the gate that blocks the whole sim2sim pipeline: if the inverse of the
feature map is not faithful, nothing downstream is meaningful. Joints must be
recovered exactly; root orientation and position within an integration-drift
tolerance (which we assert is small and also print for the record).
"""
import os
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# import the *forward* feature map from stage2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "stage2"))
from export_g1_motion import build_features  # noqa: E402

from stage3_sim2sim.decode_to_qpos36 import invert_build_features  # noqa: E402
from stage3_sim2sim.rotation_utils import yaw_yup, yaw_zup  # noqa: E402


def _synthetic_clip(T=128, dt=1 / 30, seed=0):
    """Smooth, physically-plausible root motion + joints (z-up world)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) * dt
    root_pos = np.stack([0.6 * np.sin(0.5 * t),
                         0.3 * np.cos(0.4 * t),
                         0.78 + 0.04 * np.sin(0.9 * t)], axis=1)
    yaw = 0.3 * np.sin(0.3 * t) + 0.1 * t
    pitch = 0.08 * np.sin(0.7 * t)
    roll = 0.05 * np.cos(0.6 * t)
    R = Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=1))
    quat_xyzw = R.as_quat()
    freqs = rng.uniform(0.2, 1.2, 29)
    phase = rng.uniform(0, 2 * np.pi, 29)
    joints = (0.25 * np.sin(freqs[None] * t[:, None] + phase[None])).astype(np.float32)
    return root_pos, quat_xyzw, joints, dt


def _geodesic_deg(Ra, Rb):
    rel = np.einsum("tij,tkj->tik", Ra, Rb)  # Ra Rb^T
    tr = np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


@pytest.mark.parametrize("to_yup", [False, True])
def test_build_features_roundtrip(to_yup, capsys):
    root_pos, quat_xyzw, joints, dt = _synthetic_clip()
    feats = build_features(root_pos, quat_xyzw, joints, dt, to_yup=to_yup)
    assert feats.shape == (root_pos.shape[0], 41)

    # seed heading/position from the true initial pose (the unobserved DOFs)
    R0 = Rotation.from_quat(quat_xyzw).as_matrix()
    if to_yup:
        from stage3_sim2sim.rotation_utils import T_ZUP_TO_YUP
        R0y = T_ZUP_TO_YUP @ R0 @ T_ZUP_TO_YUP.T
        yaw0 = float(yaw_yup(R0y)[0])
    else:
        yaw0 = float(yaw_zup(R0)[0])

    rp, q, j = invert_build_features(feats, dt, to_yup=to_yup,
                                     root_pos0=root_pos[0], yaw0=yaw0)

    # joints: exact passthrough
    jerr = np.abs(j - joints).max()
    # orientation
    R_rec = Rotation.from_quat(q).as_matrix()
    ang = _geodesic_deg(R_rec, R0)
    # position
    perr = np.linalg.norm(rp - root_pos, axis=1)

    with capsys.disabled():
        print(f"\n[roundtrip to_yup={to_yup}] joint max={jerr:.2e}  "
              f"orient deg mean/max={ang.mean():.3f}/{ang.max():.3f}  "
              f"root m mean/max={perr.mean():.4f}/{perr.max():.4f}")

    assert jerr < 1e-4, f"joints not recovered: {jerr}"
    assert ang.max() < 1.0, f"orientation drift too large: {ang.max():.3f} deg"
    assert perr.max() < 0.05, f"root drift too large: {perr.max():.4f} m"
