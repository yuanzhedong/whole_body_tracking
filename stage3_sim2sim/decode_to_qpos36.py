"""Inverse of the 41-D feature map: features[T,41] -> G1 ``qpos_36``.

``qpos_36 = [root_pos(3, z-up), root_quat_wxyz(4, z-up), joint_pos(29)]`` -- the
exact format the OMG HoloMotion stack (``export_holomotion_deployment_clip``)
consumes. This is the only new component in the sim2sim pipeline; everything
downstream (HoloMotion tracker + MuJoCo) already exists and is validated.

Why this needs integration (and why we report drift): the forward map stores
only the *heading-stripped* orientation (root_rot6d), *local* root velocities,
and joints. Absolute heading and absolute root position are therefore rebuilt by
**integrating** the angular/linear velocities, which accumulates drift over a
window. Pitch/roll (what matters for balance) come exactly from root_rot6d; only
heading + world translation are integrated.

Two entry points:
  * ``invert_build_features`` -- exact inverse of ``build_features`` as a pure
    function (unit-tested by a direct synthetic round-trip).
  * ``features_to_qpos36`` -- inverse of the dataset-generation path
    (``process_clip`` with ``to_yup=True``), returning true z-up ``qpos_36``.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

from .rotation_utils import (
    T_ZUP_TO_YUP, T_YUP_TO_ZUP, roty, rotz, rot6d_to_matrix,
    integrate_yaw, integrate_position, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw,
)

ROT6D = slice(0, 6)
LIN_VEL = slice(6, 9)
ANG_VEL = slice(9, 12)
JOINTS = slice(12, 41)


def _apply_basis(T, pos, R):
    """pos_new = T @ pos ; R_new = T @ R @ T^T  (same as forward apply_basis)."""
    pos_new = (T @ pos.T).T
    R_new = np.einsum("ij,tjk,lk->til", T, R, T)
    return pos_new, R_new


def invert_build_features(feats, dt, to_yup=False, root_pos0=None, yaw0=0.0):
    """Exact inverse of ``stage2.export_g1_motion.build_features``.

    Given features produced by ``build_features(root_pos, root_quat_xyzw,
    joint_pos, dt, to_yup)``, recover ``(root_pos, root_quat_xyzw, joint_pos)``
    in the SAME frame/convention that ``build_features`` received as input.

    ``root_pos0`` (3,) seeds the position integration in that input frame
    (defaults to origin at nominal height). ``yaw0`` seeds heading.
    """
    feats = np.asarray(feats, dtype=np.float64)
    joints = feats[:, JOINTS].astype(np.float32)
    R_canon = rot6d_to_matrix(feats[:, ROT6D])           # working (post-basis) frame
    lin_vel_local = feats[:, LIN_VEL]
    ang_vel_local = feats[:, ANG_VEL]

    up = 1 if to_yup else 2                               # heading axis: +y (yup) / +z (zup)
    strip = roty if to_yup else rotz
    yaw = integrate_yaw(ang_vel_local[:, up], dt, yaw0)
    R_strip = strip(yaw)                                 # = inverse of R_strip_inv used in forward
    R_work = np.einsum("tij,tjk->tik", R_strip, R_canon)  # full orientation, working frame

    lin_vel_w = np.einsum("tij,tj->ti", R_strip, lin_vel_local)  # back to working-frame world vel
    if root_pos0 is None:
        root_pos0 = np.zeros(3)
    # seed is given in build_features' INPUT frame; move to working frame for integration
    pos0_work = (T_ZUP_TO_YUP @ np.asarray(root_pos0, float)) if to_yup else np.asarray(root_pos0, float)
    pos_work = integrate_position(lin_vel_w, dt, pos0_work)

    if to_yup:  # undo the forward's apply_basis(T_ZUP_TO_YUP, ...) to return to input frame
        root_pos, R_in = _apply_basis(T_YUP_TO_ZUP, pos_work, R_work)
    else:
        root_pos, R_in = pos_work, R_work

    quat_xyzw = Rotation.from_matrix(R_in).as_quat().astype(np.float32)
    return root_pos.astype(np.float32), quat_xyzw, joints


def qpos36_to_features(qpos36, dt, double_yup=False):
    """Forward replica of the dataset feature build -> 41-D features.

    Default (``double_yup=False``) matches the FIXED ``process_clip``: a single
    z-up->y-up conversion inside ``build_features(to_yup=True)``. Set
    ``double_yup=True`` only to reproduce the legacy (buggy) double-conversion that
    older datasets / VAE checkpoints were built with.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2"))
    from export_g1_motion import build_features  # local import to avoid hard dep at import time

    qpos36 = np.asarray(qpos36, dtype=np.float64)
    root_pos = qpos36[:, 0:3]
    quat_xyzw = quat_wxyz_to_xyzw(qpos36[:, 3:7])
    joints = qpos36[:, 7:36].astype(np.float32)
    if double_yup:
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        root_pos, R = _apply_basis(T_ZUP_TO_YUP, root_pos, R)
        quat_xyzw = Rotation.from_matrix(R).as_quat()
    return build_features(root_pos, quat_xyzw, joints, dt, to_yup=True)


def features_to_qpos36(feats, dt, root_pos0_zup=None, yaw0=0.0, double_yup=False):
    """Inverse of the dataset feature build -> z-up ``qpos_36``.

    Default (``double_yup=False``) matches the FIXED single-conversion
    ``process_clip``: root height + tilt + joints are recovered exactly; absolute
    XY/heading is integrated (translation/heading-invariant by design, which the
    HoloMotion tracker tolerates). Set ``double_yup=True`` only for legacy
    double-conversion datasets (where world-root height is corrupted).

    Returns ``qpos_36[T,36] = [root_pos(3), root_quat_wxyz(4), joints(29)]``.
    """
    if root_pos0_zup is None:
        root_pos0_zup = np.zeros(3)
    # process_clip pre-converts z-up -> y-up before build_features(to_yup=True), a
    # *double* basis change. Pre-multiply the seed by T_ZUP_TO_YUP so it lands in the
    # correct (double-converted) working frame; the double_yup block then undoes it once.
    seed = (T_ZUP_TO_YUP @ np.asarray(root_pos0_zup, float)) if double_yup else np.asarray(root_pos0_zup, float)
    root_pos, quat_xyzw, joints = invert_build_features(
        feats, dt, to_yup=True, root_pos0=seed, yaw0=yaw0)
    if double_yup:
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        root_pos, R = _apply_basis(T_YUP_TO_ZUP, root_pos, R)
        quat_xyzw = Rotation.from_matrix(R).as_quat().astype(np.float32)
    quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)
    return np.concatenate([root_pos, quat_wxyz, joints], axis=1).astype(np.float32)
