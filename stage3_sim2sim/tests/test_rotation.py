"""L0 unit tests for the rotation / integration helpers."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stage3_sim2sim.rotation_utils import (
    rot6d_to_matrix, matrix_to_rot6d, roty, rotz, yaw_yup, yaw_zup,
    integrate_yaw, integrate_position,
)


def _random_rotations(n, seed=0):
    return Rotation.random(n, random_state=seed).as_matrix()


def test_rot6d_matrix_roundtrip():
    R = _random_rotations(50)
    r6 = matrix_to_rot6d(R)
    R2 = rot6d_to_matrix(r6)
    assert np.allclose(R, R2, atol=1e-5)


def test_rot6d_to_matrix_is_proper_rotation():
    r6 = matrix_to_rot6d(_random_rotations(20))
    R = rot6d_to_matrix(r6)
    eye = np.einsum("tij,tkj->tik", R, R)  # R R^T
    assert np.allclose(eye, np.eye(3)[None], atol=1e-6)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_rot6d_to_matrix_reorthonormalizes_offmanifold():
    # perturb a valid 6D off the manifold; result must still be a proper rotation
    r6 = matrix_to_rot6d(_random_rotations(10)).astype(np.float64)
    r6 += 0.05 * np.random.default_rng(1).standard_normal(r6.shape)
    R = rot6d_to_matrix(r6)
    assert np.allclose(np.einsum("tij,tkj->tik", R, R), np.eye(3)[None], atol=1e-6)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_roty_rotz_yaw_recovery():
    a = np.linspace(-2.0, 2.0, 17)
    assert np.allclose(yaw_yup(roty(a)), a, atol=1e-6)
    assert np.allclose(yaw_zup(rotz(a)), a, atol=1e-6)


def test_integrate_yaw_recovers_known_signal():
    dt = 1 / 30
    T = 200
    t = np.arange(T) * dt
    yaw_true = 0.4 * np.sin(0.5 * t) + 0.2 * t
    # finite-diff rate as the forward map would produce it (increment k-1 -> k)
    rate = np.zeros(T)
    rate[1:] = (yaw_true[1:] - yaw_true[:-1]) / dt
    yaw_rec = integrate_yaw(rate, dt, yaw0=yaw_true[0])
    assert np.allclose(yaw_rec, yaw_true, atol=1e-9)


def test_integrate_position_recovers_smooth_trajectory():
    dt = 1 / 30
    t = np.arange(150) * dt
    pos = np.stack([np.sin(t), 0.5 * np.cos(0.7 * t), 0.8 + 0.1 * t], axis=1)
    vel = np.gradient(pos, dt, axis=0)  # exactly what the forward map stores
    rec = integrate_position(vel, dt, pos0=pos[0])
    # trapezoid inverting central-difference: small bounded error, not exact
    err = np.linalg.norm(rec - pos, axis=1).max()
    assert err < 5e-3, f"position integration drift too large: {err:.4f} m"
