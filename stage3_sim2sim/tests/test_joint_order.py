"""Tests for the G1 joint-order conventions + the feature<->OMG mapping.

Guards the bug where feature-order joints were rendered with the OMG/MuJoCo
renderer (which expects a different order) -> scrambled pose. These pin both
orderings, verify the permutation, and detect drift from the source definitions.
"""
import os
import sys
import numpy as np
import pytest

from stage3_sim2sim.joint_order import (
    FEATURE_ORDER, OMG_ORDER, FEATURE_TO_OMG, OMG_TO_FEATURE,
    feature_to_omg, qpos36_feature_to_omg,
)


def test_orders_are_same_29_joint_set():
    assert len(FEATURE_ORDER) == 29 and len(OMG_ORDER) == 29
    assert set(FEATURE_ORDER) == set(OMG_ORDER)          # same joints
    assert len(set(FEATURE_ORDER)) == 29                 # no duplicates


def test_orders_actually_differ():
    # the whole point: the two conventions are NOT the same ordering
    assert FEATURE_ORDER != OMG_ORDER


def test_perms_are_bijections_and_inverses():
    assert sorted(FEATURE_TO_OMG.tolist()) == list(range(29))
    assert sorted(OMG_TO_FEATURE.tolist()) == list(range(29))
    x = np.arange(29)
    assert np.array_equal(x[FEATURE_TO_OMG][OMG_TO_FEATURE], x)   # round-trip identity


def test_feature_to_omg_maps_names_correctly():
    mapped = [FEATURE_ORDER[i] for i in FEATURE_TO_OMG]
    assert mapped == OMG_ORDER


def test_feature_to_omg_on_array():
    j = np.arange(29.0)[None]                 # feature-order values = their indices
    out = feature_to_omg(j)[0]
    # out[i] should be the feature-index of OMG_ORDER[i]
    assert np.array_equal(out, FEATURE_TO_OMG.astype(float))


def test_qpos36_reorder_preserves_root_and_roundtrips():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((10, 36))
    q2 = qpos36_feature_to_omg(q)
    assert np.array_equal(q2[:, :7], q[:, :7])               # root untouched
    back = q2.copy(); back[:, 7:36] = q2[:, 7:36][:, OMG_TO_FEATURE]
    assert np.allclose(back, q)                              # joints round-trip


def test_feature_order_matches_export_source():
    """Drift guard: FEATURE_ORDER must equal stage2/export_g1_motion.JOINT_NAMES."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "stage2"))
    try:
        from export_g1_motion import JOINT_NAMES
    except Exception:
        pytest.skip("export_g1_motion not importable")
    assert list(JOINT_NAMES) == FEATURE_ORDER


def test_omg_order_matches_omg_source():
    """Drift guard: OMG_ORDER must equal omg.robots.g1.constants.G1_JOINT_NAMES."""
    sys.path.insert(0, "/ws/user/yzdong/src/github/OMG/src")
    try:
        from omg.robots.g1.constants import G1_JOINT_NAMES
    except Exception:
        pytest.skip("omg package not importable")
    assert list(G1_JOINT_NAMES) == OMG_ORDER
