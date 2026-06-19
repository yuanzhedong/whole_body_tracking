"""Corner-case + known-answer tests for the feature map, inverse, hybrid, metrics.

Known-answer: feed inputs whose correct output is analytically obvious (static,
pure-yaw, pure-translation, quat double-cover) and check the exact values.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "stage2"))
from export_g1_motion import build_features  # noqa: E402

from stage3_sim2sim.decode_to_qpos36 import invert_build_features  # noqa: E402
from stage3_sim2sim.rotation_utils import rotz  # noqa: E402
from stage3_sim2sim.sim2sim import rollout_metrics, build_hybrid_qpos36  # noqa: E402

DT = 1 / 30
IDENT_ROT6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)


# ---------------- C2 forward feature map: known answers ----------------
def test_forward_static_clip():
    """No motion: velocities ~0, rot6d = identity columns, joints passthrough."""
    T = 30
    root = np.tile([0.1, 0.2, 0.78], (T, 1)).astype(float)
    quat = np.tile([0, 0, 0, 1], (T, 1)).astype(float)   # xyzw identity
    joints = np.tile(np.linspace(-0.3, 0.3, 29), (T, 1))
    f = build_features(root, quat, joints, DT, to_yup=False)
    assert np.allclose(f[:, 0:6], IDENT_ROT6D, atol=1e-6)        # identity orientation
    assert np.allclose(f[:, 6:9], 0, atol=1e-6)                  # lin vel 0
    assert np.allclose(f[:, 9:12], 0, atol=1e-6)                 # ang vel 0
    assert np.allclose(f[:, 12:41], joints, atol=1e-6)           # joints exact


def test_forward_pure_yaw():
    """Constant yaw rate about +z: heading stripped (rot6d const), ang_vel only on z."""
    T = 40
    w = 0.5  # rad/s
    t = np.arange(T) * DT
    R = rotz(w * t)
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(R).as_quat()
    root = np.tile([0, 0, 0.78], (T, 1)).astype(float)
    joints = np.zeros((T, 29))
    f = build_features(root, quat, joints, DT, to_yup=False)
    assert np.allclose(f[2:-2, 0:6], IDENT_ROT6D, atol=1e-4)     # canon orientation constant
    assert np.allclose(f[2:-2, 6:9], 0, atol=1e-4)              # no translation
    assert np.allclose(f[2:-2, 9], 0, atol=1e-4) and np.allclose(f[2:-2, 10], 0, atol=1e-4)
    assert np.allclose(f[2:-2, 11], w, atol=1e-3)               # yaw rate on +z


def test_forward_pure_translation():
    """Constant world velocity, identity orientation: lin_vel = velocity, rest ~0."""
    T = 30
    vel = np.array([0.3, -0.2, 0.0])
    root = (np.arange(T)[:, None] * DT) * vel + np.array([0, 0, 0.78])
    quat = np.tile([0, 0, 0, 1], (T, 1)).astype(float)
    f = build_features(root, quat, np.zeros((T, 29)), DT, to_yup=False)
    assert np.allclose(f[2:-2, 6:9], vel, atol=1e-4)
    assert np.allclose(f[2:-2, 9:12], 0, atol=1e-4)
    assert np.allclose(f[2:-2, 0:6], IDENT_ROT6D, atol=1e-6)


def test_forward_quat_double_cover():
    """q and -q are the same rotation -> identical features."""
    rng = np.random.default_rng(0)
    T = 20
    from scipy.spatial.transform import Rotation
    quat = Rotation.random(T, random_state=1).as_quat()
    root = rng.standard_normal((T, 3))
    joints = rng.standard_normal((T, 29))
    f1 = build_features(root, quat, joints, DT, to_yup=False)
    f2 = build_features(root, -quat, joints, DT, to_yup=False)
    assert np.allclose(f1, f2, atol=1e-5)


# ---------------- C3 inverse corner cases ----------------
def test_inverse_zero_velocity_constant_root():
    """Zero lin/ang velocity features -> root stays at the seed position."""
    T = 30
    feats = np.zeros((T, 41), dtype=np.float32)
    feats[:, 0:6] = IDENT_ROT6D                  # identity orientation
    seed = np.array([1.0, 2.0, 0.78])
    rp, q, j = invert_build_features(feats, DT, to_yup=False, root_pos0=seed)
    assert np.allclose(rp, seed, atol=1e-6)      # no drift with zero velocity


def test_inverse_short_window_no_crash():
    feats = np.zeros((4, 41), dtype=np.float32)
    feats[:, 0:6] = IDENT_ROT6D
    rp, q, j = invert_build_features(feats, DT, to_yup=False, root_pos0=np.zeros(3))
    assert rp.shape == (4, 3) and q.shape == (4, 4) and j.shape == (4, 29)


def test_inverse_yaw_wraparound_continuous():
    """Yaw integrating through +/-pi must give a continuous rotation (no quat blowup)."""
    T = 200
    feats = np.zeros((T, 41), dtype=np.float32)
    feats[:, 0:6] = IDENT_ROT6D
    feats[:, 11] = 3.0  # large +z yaw rate -> sweeps several pi over the window
    rp, q, j = invert_build_features(feats, DT, to_yup=False, root_pos0=np.zeros(3))
    assert np.all(np.isfinite(q))
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)  # unit quaternions throughout


# ---------------- C5 hybrid + C9 metrics corner cases ----------------
def test_hybrid_length_mismatch_truncates():
    feats = np.zeros((100, 41), dtype=np.float32); feats[:, 12:41] = 0.5
    orig = np.zeros((80, 36), dtype=np.float32); orig[:, 3] = 1.0
    q = build_hybrid_qpos36(feats, orig)
    assert q.shape == (80, 36)               # min length
    assert np.allclose(q[:, 7:36], 0.5)


def test_metrics_length_mismatch_uses_min():
    ex = np.zeros((100, 36), dtype=np.float32); ex[:, 2] = 0.75; ex[:, 3] = 1.0
    ref = np.zeros((70, 36), dtype=np.float32); ref[:, 2] = 0.75; ref[:, 3] = 1.0
    m = rollout_metrics(ex, ref)
    assert m["n_frames"] == 70 and m["survival"] == 1.0


def test_metrics_single_frame():
    ex = np.array([[0, 0, 0.75] + [0, 0, 0, 0] + [0] * 29], dtype=np.float32)
    ex[0, 3] = 1.0
    m = rollout_metrics(ex, ex)
    assert m["n_frames"] == 1 and m["survival"] == 1.0
