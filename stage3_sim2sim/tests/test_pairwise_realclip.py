"""L1 pairwise: inverse on REAL seed clips (not synthetic) + drift report.

Validates that `features_to_qpos36` is a faithful inverse of the dataset feature
build on the real feature distribution: joints pass through exactly, and a
feature-space round-trip (features -> qpos36 -> features') recovers the root
feature blocks within an integration-drift tolerance. Skips cleanly if the seed
dataset isn't present on this machine.
"""
import glob
import numpy as np
import pytest

from stage3_sim2sim.decode_to_qpos36 import features_to_qpos36, qpos36_to_features

VAL_DIR = "/scratch/user/yzdong/OMG-Data/umt/g1_seed_full_yup/val"
WS = 128


def _val_clips(n=5):
    files = sorted(glob.glob(f"{VAL_DIR}/*.npz"))
    if not files:
        pytest.skip(f"seed val set not present at {VAL_DIR}")
    return files[:n]


def test_joint_passthrough_exact():
    for f in _val_clips():
        d = np.load(f, allow_pickle=True)
        feats = d["motion"][:WS]
        if feats.shape[0] < WS:
            continue
        dt = 1.0 / float(np.asarray(d["fps"]).reshape(-1)[0])
        qpos = features_to_qpos36(feats, dt)
        # qpos joints (cols 7:36) must equal the feature joints (cols 12:41) exactly
        assert np.allclose(qpos[:, 7:36], feats[:, 12:41], atol=1e-5)
        # ...and equal the stored raw joint_pos exactly
        assert np.allclose(qpos[:, 7:36], d["joint_pos"][:WS], atol=1e-4)


def test_feature_space_roundtrip_realclips(capsys):
    """features -> qpos36 -> features' on real clips.

    Asserts the quantities that form the robot reference (joints, tilt via rot6d,
    linear velocity). The angular-velocity block is only *reported*: it is a
    finite-difference-derived feature, NOT part of qpos_36 (the tracker computes
    its own velocities from the qpos sequence), so a tight round-trip of it is
    neither expected nor required.
    """
    rot_errs, lin_errs, ang_errs, jt_errs = [], [], [], []
    for f in _val_clips(8):
        d = np.load(f, allow_pickle=True)
        feats = d["motion"][:WS].astype(np.float64)
        if feats.shape[0] < WS:
            continue
        dt = 1.0 / float(np.asarray(d["fps"]).reshape(-1)[0])
        qpos = features_to_qpos36(feats, dt)
        feats2 = qpos36_to_features(qpos, dt)
        # interior frames (exclude 2-frame finite-difference boundary)
        sl = slice(2, -2)
        rot_errs.append(np.abs(feats2[sl, 0:6] - feats[sl, 0:6]).max())
        lin_errs.append(np.abs(feats2[sl, 6:9] - feats[sl, 6:9]).max())
        ang_errs.append(np.abs(feats2[sl, 9:12] - feats[sl, 9:12]).max())
        jt_errs.append(np.abs(feats2[sl, 12:41] - feats[sl, 12:41]).max())
    rot, lin, ang, jt = (float(np.mean(x)) for x in (rot_errs, lin_errs, ang_errs, jt_errs))
    with capsys.disabled():
        print(f"\n[L1 real-clip feature round-trip, interior] joints max={jt:.2e}  "
              f"rot6d(tilt) max={rot:.2e}  lin_vel max={lin:.2e}  "
              f"ang_vel max={ang:.3f} (derived, reported only)")
    # what forms the qpos_36 reference must be faithful:
    assert jt < 1e-4, f"joints not exact: {jt}"
    assert rot < 1e-3, f"tilt (rot6d) drifted: {rot}"
    assert lin < 5e-3, f"linear velocity drifted: {lin}"


def test_root_drift_report(capsys):
    """Report integrated-root vs stored ground-truth root drift (y-up body_pos_w[:,0])."""
    drifts = []
    for f in _val_clips(8):
        d = np.load(f, allow_pickle=True)
        feats = d["motion"][:WS].astype(np.float64)
        if feats.shape[0] < WS:
            continue
        dt = 1.0 / float(np.asarray(d["fps"]).reshape(-1)[0])
        # seed at the true initial root (y-up body_pos_w idx0 -> z-up via inverse basis)
        from stage3_sim2sim.rotation_utils import T_YUP_TO_ZUP
        root0_zup = T_YUP_TO_ZUP @ d["body_pos_w"][0, 0]
        qpos = features_to_qpos36(feats, dt, root_pos0_zup=root0_zup)
        # ground-truth root path in z-up
        gt_zup = (T_YUP_TO_ZUP @ d["body_pos_w"][:WS, 0].T).T
        # compare horizontal drift relative to start (heading is unobserved, so this is an upper bound)
        rec_rel = qpos[:, 0:3] - qpos[0, 0:3]
        gt_rel = gt_zup - gt_zup[0]
        drifts.append(np.linalg.norm(rec_rel - gt_rel, axis=1).max())
    with capsys.disabled():
        print(f"\n[L1 root-drift] mean/max over clips: "
              f"{np.mean(drifts):.3f} / {np.max(drifts):.3f} m (128-frame window)")
    # informational gate: drift should be sub-metre over ~4 s; hard fail only if absurd
    assert np.max(drifts) < 1.0
