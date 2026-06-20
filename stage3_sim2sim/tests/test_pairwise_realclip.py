"""L1 pairwise: forward+inverse on REAL seed clips (FIXED single-conversion convention).

Source of truth is the artifact ground-truth qpos (build_qpos36_from_artifact); we
build features with the fixed forward (qpos36_to_features, single y-up conversion) and
invert them. Joints, root TILT, and root HEIGHT are recovered exactly; absolute
XY/heading is integrated (translation/heading-invariant by design) and only reported.
"""
import glob
import numpy as np
import pytest

from stage3_sim2sim.decode_to_qpos36 import features_to_qpos36, qpos36_to_features
from stage3_sim2sim.sim2sim import build_qpos36_from_artifact

ART_GLOB = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/artifacts_seed_full/*walk*/motion.npz"
WS = 128
DT = 1 / 30


def _clips(n=6):
    arts = sorted(glob.glob(ART_GLOB))
    if not arts:
        pytest.skip("seed artifacts not present")
    out = []
    for a in arts:
        q = build_qpos36_from_artifact(a)
        if q.shape[0] >= WS:
            out.append(q)
        if len(out) >= n:
            break
    if not out:
        pytest.skip("no clips >= window length")
    return out


def test_joint_passthrough_exact():
    for q in _clips():
        feats = qpos36_to_features(q, DT)
        rec = features_to_qpos36(feats, DT, root_pos0_zup=q[0, :3])
        assert np.allclose(rec[:, 7:36], feats[:, 12:41], atol=1e-5)   # qpos joints == feature joints
        assert np.allclose(rec[:, 7:36], q[:len(rec), 7:36], atol=1e-4)  # == ground-truth joints


def test_absolute_root_height_recovered():
    """FIXED: single-conversion forward -> the inverse recovers absolute pelvis height."""
    for q in _clips():
        feats = qpos36_to_features(q, DT)
        rec = features_to_qpos36(feats, DT, root_pos0_zup=q[0, :3])
        n = min(len(rec), len(q))
        zerr = np.abs(rec[:n, 2] - q[:n, 2]).max()
        assert zerr < 0.02, f"absolute pelvis height not recovered: {zerr:.4f} m"


def test_feature_space_roundtrip(capsys):
    rot_e, lin_e, ang_e, jt_e = [], [], [], []
    for q in _clips(8):
        feats = qpos36_to_features(q, DT).astype(np.float64)
        rec = features_to_qpos36(feats, DT, root_pos0_zup=q[0, :3])
        feats2 = qpos36_to_features(rec, DT)
        sl = slice(2, -2)
        rot_e.append(np.abs(feats2[sl, 0:6] - feats[sl, 0:6]).max())
        lin_e.append(np.abs(feats2[sl, 6:9] - feats[sl, 6:9]).max())
        ang_e.append(np.abs(feats2[sl, 9:12] - feats[sl, 9:12]).max())
        jt_e.append(np.abs(feats2[sl, 12:41] - feats[sl, 12:41]).max())
    rot, lin, ang, jt = (float(np.mean(x)) for x in (rot_e, lin_e, ang_e, jt_e))
    with capsys.disabled():
        print(f"\n[L1 feature round-trip] joints={jt:.2e} tilt={rot:.2e} lin_vel={lin:.2e} "
              f"ang_vel={ang:.3f} (derived, reported)")
    assert jt < 1e-4 and rot < 1e-3 and lin < 5e-3


def test_root_xy_drift_reported(capsys):
    """Absolute XY/heading is integrated and NOT preserved (heading-canonical rep);
    reported only. The tracker is XY/heading-invariant, so this does not affect sim2sim."""
    heights, xys = [], []
    for q in _clips(8):
        feats = qpos36_to_features(q, DT)
        rec = features_to_qpos36(feats, DT, root_pos0_zup=q[0, :3])
        n = min(len(rec), len(q))
        heights.append(np.abs(rec[:n, 2] - q[:n, 2]).max())
        xys.append(np.linalg.norm(rec[:n, :2] - q[:n, :2], axis=1).max())
    with capsys.disabled():
        print(f"\n[L1 root] height err max={max(heights):.4f}m (recovered)  "
              f"XY drift max={max(xys):.2f}m (heading-invariant, expected)")
    assert max(heights) < 0.02   # height IS recovered; XY drift is expected and not asserted
