"""Format/contract tests for hf_to_csv.py. CPU-only, no Isaac, no network (except the
optional real-file test, which is skipped if the clip hasn't been downloaded)."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hf_to_csv as h  # noqa: E402


def _synthetic(n=50):
    """A valid-looking [n,36] clip: zero-ish root, unit quats, mid-range joints."""
    rng = np.random.RandomState(0)
    arr = np.zeros((n, h.N_COLS), dtype=np.float64)
    arr[:, 0] = np.linspace(0, 1.0, n)      # x travel
    arr[:, 2] = 0.0                         # stored z ~ 0 (offset applied later)
    arr[:, 3:7] = [0.0, 0.0, 0.0, 1.0]      # identity quat xyzw
    for i, (_, lo, hi) in enumerate(h.JOINT_LIMITS):
        arr[:, 7 + i] = 0.5 * (lo + hi)     # mid-range -> within limits
    return arr


def test_parse_fps():
    assert h.parse_fps_from_filename("g1/KIT/425/walking_01_poses_100_jpos.npy") == 100
    assert h.parse_fps_from_filename("g1/ACCAD/s007/QkWalk1_poses_120_jpos.npy") == 120
    with pytest.raises(ValueError):
        h.parse_fps_from_filename("nonsense.npy")


def test_height_offset_lifts_pelvis():
    arr = _synthetic()
    out = h.apply_height_offset(arr, 0.793)
    assert np.allclose(out[:, 2], 0.793)          # z lifted to nominal pelvis height
    assert np.allclose(out[:, :2], arr[:, :2])    # xy untouched
    assert np.allclose(out[:, 3:], arr[:, 3:])    # quat + joints untouched


def test_validate_shape_and_quat():
    rep = h.validate(h.apply_height_offset(_synthetic()))
    assert rep["frames"] == 50
    assert rep["quat_norm_max_dev"] < 1e-6
    assert rep["joint_limit_viol_frac_any"] == 0.0
    assert 0.79 <= rep["root_z_min"] <= 0.80


def test_validate_rejects_bad_shape():
    with pytest.raises(ValueError):
        h.validate(np.zeros((10, 30)))


def test_validate_rejects_nan():
    arr = h.apply_height_offset(_synthetic())
    arr[5, 10] = np.nan
    with pytest.raises(ValueError):
        h.validate(arr)


def test_validate_flags_joint_violation():
    arr = h.apply_height_offset(_synthetic())
    arr[:, 7 + 3] = 99.0  # left_knee way over its upper limit
    rep = h.validate(arr)
    assert rep["joint_limit_viol_frac_any"] == 1.0
    assert rep["worst_joints"][0][0] == "left_knee_joint"


def test_write_csv_roundtrip(tmp_path):
    arr = h.apply_height_offset(_synthetic())
    out = str(tmp_path / "m.csv")
    h.write_csv(arr, out)
    back = np.loadtxt(out, delimiter=",")
    assert back.shape == arr.shape
    assert np.allclose(back, arr, atol=1e-6)


# ---- optional: validate the actual downloaded clip if present ----
_REAL = "retargeting/data/g1"


def _find_real_npy():
    for root, _, files in os.walk(_REAL):
        for f in files:
            if f.endswith(".npy"):
                return os.path.join(root, f)
    return None


@pytest.mark.skipif(_find_real_npy() is None, reason="no downloaded clip present")
def test_real_clip_contract():
    path = _find_real_npy()
    arr = h.apply_height_offset(np.load(path))
    rep = h.validate(arr)
    assert rep["quat_norm_max_dev"] < 1e-2          # real data: near-unit quats
    assert 0.5 < rep["root_z_min"] < 1.2            # pelvis at a plausible standing height
    assert rep["joint_limit_viol_frac_any"] < 0.10  # mostly within limits
