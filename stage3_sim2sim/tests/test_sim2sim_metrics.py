"""L0 unit tests for rollout scoring + artifact qpos builder (no heavy deps)."""
import numpy as np
import pytest

from stage3_sim2sim.sim2sim import (
    rollout_metrics, build_qpos36_from_artifact, build_hybrid_qpos36, FALL_HEIGHT,
)


def _qpos(root_z, T=100):
    q = np.zeros((T, 36), dtype=np.float32)
    q[:, 2] = root_z
    q[:, 3] = 1.0  # wxyz identity
    return q


def test_survival_upright():
    ex = _qpos(0.75)
    m = rollout_metrics(ex, ex)
    assert m["survival"] == 1.0
    assert m["joint_rmse_deg"] == 0.0
    assert m["root_xy_drift_max"] == 0.0


def test_survival_fallen():
    ex = _qpos(0.20)            # collapsed below fall height
    ref = _qpos(0.75)
    m = rollout_metrics(ex, ref)
    assert m["survival"] == 0.0
    assert m["root_z_min"] < FALL_HEIGHT


def test_survival_partial_fall():
    ex = _qpos(0.75)
    ex[50:, 2] = 0.2            # falls halfway through
    m = rollout_metrics(ex, _qpos(0.75))
    assert m["survival"] == pytest.approx(0.5, abs=0.02)


def test_joint_rmse_known_offset():
    ref = _qpos(0.75)
    ex = ref.copy()
    ex[:, 7:36] += np.deg2rad(10.0)   # uniform 10 deg joint offset
    m = rollout_metrics(ex, ref)
    assert m["joint_rmse_deg"] == pytest.approx(10.0, abs=1e-3)


def test_root_drift_known():
    ref = _qpos(0.75)
    ex = ref.copy()
    ex[:, 0] += 0.3                    # 0.3 m x offset
    m = rollout_metrics(ex, ref)
    assert m["root_xy_drift_max"] == pytest.approx(0.3, abs=1e-4)


def test_build_hybrid_qpos36():
    T = 50
    orig = np.zeros((T, 36), dtype=np.float32)
    orig[:, 2] = 0.75            # original root height
    orig[:, 3] = 1.0
    orig[:, 7:36] = 0.1          # original joints
    feats = np.zeros((T, 41), dtype=np.float32)
    feats[:, 12:41] = 0.4        # decoded joints
    q = build_hybrid_qpos36(feats, orig)
    assert np.allclose(q[:, :7], orig[:, :7])      # root pose from original
    assert np.allclose(q[:, 7:36], 0.4)            # joints from decoded features


def test_build_qpos36_from_artifact_shape(tmp_path):
    T = 20
    art = tmp_path / "motion.npz"
    rng = np.random.default_rng(0)
    body_pos = rng.standard_normal((T, 30, 3)).astype(np.float32)
    body_pos[:, 0, 2] = 0.75
    body_quat = np.tile([1, 0, 0, 0], (T, 30, 1)).astype(np.float32)
    np.savez(art, joint_pos=rng.standard_normal((T, 29)).astype(np.float32),
             body_pos_w=body_pos, body_quat_w=body_quat, fps=np.array([30.0]))
    q = build_qpos36_from_artifact(str(art))
    assert q.shape == (T, 36)
    assert np.allclose(q[:, 2], 0.75)          # root z = pelvis height
    assert np.allclose(q[:, 3:7], [1, 0, 0, 0])  # wxyz
