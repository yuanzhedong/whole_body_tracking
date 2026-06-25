"""Regression tests pinning the HoloMotion near-ground root-cause analysis.

Reads the committed holomotion_rootcause.json (produced by
bfmzero_compare/analyze_holo_rootcause.py). Skips if absent.
"""
import json
import os

import pytest

HERE = os.path.dirname(__file__)
RC = os.path.join(HERE, "..", "bfmzero_compare", "holomotion_rootcause.json")


def _load():
    if not os.path.exists(RC):
        pytest.skip("holomotion_rootcause.json not present")
    return json.load(open(RC))


def test_init_mismatch_ruled_out():
    """Robot initializes AT the reference pose -> failure is not an init mismatch."""
    h = _load()["H1_init_ruled_out"]
    assert h["init_gap_max_abs_m"] < 0.06, h


def test_failure_driven_by_posture_depth():
    """Survival correlates positively with reference pelvis height (lower -> worse)."""
    h = _load()["H2_depth_driven"]
    assert h["corr_survival_vs_ref_pelvis_min"] > 0.5, h


def test_data_feed_is_correct():
    """Positive control: on clips it survives, HoloMotion tracks faithfully through the
    identical feeding pipeline -> not a wrong-data artifact."""
    h = _load()["H3_data_correct"]
    assert h["n_success_clips"] >= 5
    assert h["success_mean_joint_err_deg"] < 25.0, h


def test_policy_under_commands_not_physics():
    """Decisive: during the descent the policy COMMANDS far shallower knee flexion than
    the reference needs, and the knee ACHIEVES at-or-beyond the command -> the actuator
    is not the limit; the policy under-commands deep flexion."""
    rows = _load()["H4_policy_not_physics"]["rows"]
    assert rows, "no H4 rows"
    knee = rows[0]["left_knee_joint"]
    # reference requires deep flexion, policy commands much less
    assert abs(knee["reference_deg"]) - abs(knee["commanded_deg"]) > 50, knee
    # achieved meets-or-exceeds the command -> not torque-limited
    assert abs(knee["achieved_deg"]) >= abs(knee["commanded_deg"]) - 5, knee
