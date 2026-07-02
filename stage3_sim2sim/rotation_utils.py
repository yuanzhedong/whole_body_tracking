"""Rotation / frame helpers shared by the sim2sim decode path.

These mirror the *exact* conventions used by the forward feature map in
``stage2/export_g1_motion.py`` so that ``decode_to_qpos36`` is a faithful inverse:

  - 6D rotation = the first two **columns** of a rotation matrix, flattened as
    ``[c0x,c0y,c0z, c1x,c1y,c1z]`` (matches ``R[:, :, :2].transpose(0,2,1).reshape(-1,6)``).
  - heading ("yaw") is about +z in z-up, about +y in y-up.
  - basis change z-up <-> y-up uses ``T_ZUP_TO_YUP`` (a proper rotation, det=+1).
"""
from __future__ import annotations
import numpy as np

# z-up (Isaac) -> y-up (OmniMM): new_x=old_x, new_y=old_z, new_z=-old_y  (det=+1)
T_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
T_YUP_TO_ZUP = T_ZUP_TO_YUP.T  # inverse of a rotation = transpose


def rotz(a):
    """Batch rotation matrices about +z by angles a[T]."""
    a = np.asarray(a, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((a.shape[0], 3, 3))
    R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1.0
    return R


def roty(a):
    """Batch rotation matrices about +y by angles a[T]."""
    a = np.asarray(a, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((a.shape[0], 3, 3))
    R[:, 0, 0] = c; R[:, 0, 2] = s; R[:, 1, 1] = 1.0; R[:, 2, 0] = -s; R[:, 2, 2] = c
    return R


def yaw_zup(R):
    """Heading about +z (z-up): atan2(R[1,0], R[0,0])."""
    return np.arctan2(R[:, 1, 0], R[:, 0, 0])


def yaw_yup(R):
    """Heading about +y (y-up): atan2(-R[2,0], R[0,0])."""
    return np.arctan2(-R[:, 2, 0], R[:, 0, 0])


def matrix_to_rot6d(R):
    """[T,3,3] -> [T,6] = first two columns, row-major (c0, c1). Inverse of rot6d_to_matrix."""
    return R[:, :, :2].transpose(0, 2, 1).reshape(-1, 6).astype(np.float32)


def rot6d_to_matrix(rot6d):
    """[T,6] -> [T,3,3] via Gram-Schmidt on the two stored columns.

    rot6d is laid out as ``[c0(3), c1(3)]`` (the first two columns of R). We
    re-orthonormalize (c0, c1) and set c2 = c0 x c1 so the result is a proper
    rotation even when the decoded 6D is slightly off-manifold.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    c0 = rot6d[:, 0:3]
    c1 = rot6d[:, 3:6]
    b0 = c0 / np.linalg.norm(c0, axis=1, keepdims=True).clip(1e-9)
    # remove b0 component from c1, normalize
    c1 = c1 - (np.sum(b0 * c1, axis=1, keepdims=True) * b0)
    b1 = c1 / np.linalg.norm(c1, axis=1, keepdims=True).clip(1e-9)
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=2)  # columns = (b0,b1,b2) -> [T,3,3]


def integrate_yaw(yaw_rate, dt, yaw0=0.0):
    """yaw(t) = yaw0 + dt * cumulative sum of yaw_rate up to (and including) frame t-1.

    Mirrors the forward map's finite-difference angular velocity: the heading
    increment from frame k-1 to k is yaw_rate[k]*dt. Frame 0 keeps yaw0.
    """
    yaw_rate = np.asarray(yaw_rate, dtype=np.float64)
    inc = np.zeros_like(yaw_rate)
    inc[1:] = yaw_rate[1:] * dt
    return yaw0 + np.cumsum(inc)


def integrate_position(vel_w, dt, pos0):
    """Exact inverse of the forward map's ``np.gradient(root_pos, dt, axis=0)``.

    ``np.gradient`` uses a forward difference at frame 0 (``v0=(p1-p0)/dt``) and
    central differences in the interior (``vk=(p[k+1]-p[k-1])/2dt``). Inverting
    those exactly recovers position with no accumulated scheme-mismatch drift:

        p[1]   = p[0] + v[0]*dt
        p[k+1] = p[k-1] + 2*dt*v[k]   (k = 1 .. T-2)

    (The even/odd index chains are tied together by p[0] and the p[1] seed, so
    there is no sawtooth/drift — unlike trapezoid integration of central diffs.)
    """
    vel_w = np.asarray(vel_w, dtype=np.float64)
    T = vel_w.shape[0]
    pos = np.zeros_like(vel_w)
    pos[0] = np.asarray(pos0, dtype=np.float64)
    if T == 1:
        return pos
    pos[1] = pos[0] + vel_w[0] * dt
    for k in range(1, T - 1):
        pos[k + 1] = pos[k - 1] + 2.0 * dt * vel_w[k]
    return pos


def quat_xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]
