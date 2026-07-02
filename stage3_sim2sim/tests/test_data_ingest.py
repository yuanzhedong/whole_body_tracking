"""C1 data-ingest correctness: seed CSV -> artifact (units, resample, joint permutation)."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "stage2"))
import seed_to_artifacts as s2a  # noqa: E402


def test_joint_permutation_is_bijection():
    perm = np.asarray(s2a.PERM)
    assert perm.shape == (29,)
    assert sorted(perm.tolist()) == list(range(29))   # a true permutation, no dup/drop


def _write_csv(tmp_path, n_rows, root_cm=(50.0, 60.0, 78.0), joint_deg=10.0):
    cols = 36
    data = np.zeros((n_rows, cols))
    data[:, 0] = np.arange(n_rows)            # frame
    data[:, 1:4] = root_cm                     # root translate (cm)
    data[:, 4:7] = 0.0                         # root euler (deg) -> identity
    data[:, 7:36] = joint_deg                  # 29 joints (deg)
    p = tmp_path / "clip.csv"
    header = ",".join([f"c{i}" for i in range(cols)])
    np.savetxt(p, data, delimiter=",", header=header, comments="")
    return str(p)


def test_convert_csv_units_and_resample(tmp_path):
    csv = _write_csv(tmp_path, n_rows=160)     # 160 @120fps -> 40 @30fps (stride 4)
    out = s2a.convert_csv(csv)
    assert out is not None
    T = out["joint_pos"].shape[0]
    assert T == 160 // s2a.STRIDE              # 40 frames
    # cm -> m on the root (body idx 0)
    assert np.allclose(out["body_pos_w"][:, 0, :], [0.5, 0.6, 0.78], atol=1e-5)
    # deg -> rad on joints (constant 10 deg)
    assert np.allclose(out["joint_pos"], np.deg2rad(10.0), atol=1e-5)
    # identity root rotation -> wxyz ~ [1,0,0,0]
    assert np.allclose(np.abs(out["body_quat_w"][:, 0, :]), [1, 0, 0, 0], atol=1e-5)
    assert float(np.asarray(out["fps"]).reshape(-1)[0]) == s2a.DST_FPS


def test_convert_csv_too_short_returns_none(tmp_path):
    csv = _write_csv(tmp_path, n_rows=40)      # below STRIDE*20 = 80 -> skipped
    assert s2a.convert_csv(csv) is None


def test_convert_csv_joint_permutation_applied(tmp_path):
    """Distinct per-joint values must be reordered OMG->UniMoTok by PERM."""
    n = 160
    data = np.zeros((n, 36)); data[:, 0] = np.arange(n)
    data[:, 1:4] = 0.0
    data[:, 7:36] = np.arange(29) * 1.0        # joint j (OMG order) = j degrees
    p = tmp_path / "c.csv"
    np.savetxt(p, data, delimiter=",", header=",".join(["c"] * 36), comments="")
    out = s2a.convert_csv(str(p))
    expected = np.deg2rad(np.arange(29))[s2a.PERM]   # OMG values reordered to UniMoTok
    assert np.allclose(out["joint_pos"][0], expected, atol=1e-6)
