"""Seed-dataset performance regression tests.

Codify the measured performance of the FINAL lat-512 VAE on the seed dataset as
thresholds, reading the committed report JSONs (so they run anywhere, no GPU):
  - reconstruction quality  -> seed_recon_report.json    (eval_seed_recon.py, 500 val clips)
  - full-root sim2sim        -> seed_sim2sim_sweep_summary.json (run_l3_eval.py --full_root, 100 clips)
Regenerate the JSONs with those scripts if the model/data changes.
"""
import json
import os
import pytest

HERE = os.path.dirname(__file__)
RECON = os.path.join(HERE, "..", "seed_recon_report.json")
SWEEP = os.path.join(HERE, "..", "seed_sim2sim_sweep_summary.json")


def _load(p):
    if not os.path.exists(p):
        pytest.skip(f"report not present: {p}")
    return json.load(open(p))


def test_reconstruction_quality_thresholds():
    r = _load(RECON)
    assert r["n_clips"] >= 100
    # body-pose reconstruction: overall and static are tight; dynamic is the known ~13 floor
    assert r["overall"]["joint_rmse_deg"] < 10.0, r["overall"]
    assert r["static"]["joint_rmse_deg"] < 6.0, r["static"]
    assert r["dynamic"]["joint_rmse_deg"] < 15.0, r["dynamic"]   # dynamic floor (architectural)
    # root orientation is reconstructed (geodesic), not garbage
    assert r["overall"]["root_geo_deg"] < 10.0, r["overall"]


def test_sim2sim_decoded_preserves_executability():
    s = _load(SWEEP)
    assert s["n"] >= 50
    # the VAE decode must not meaningfully degrade physical survival vs the ORIGINAL motion
    # (failures are tracker limits on near-ground motion, which the original hits too)
    assert s["decoded_survival_mean"] >= 0.95 * s["orig_survival_mean"], s
    # a clear majority of clips survive fully under full decoded root
    assert s["frac_full_survival"] >= 0.6, s


def test_sim2sim_decoded_survival_on_tracker_feasible():
    """Isolated from tracker limits: on clips the tracker CAN execute (original survival>=0.9),
    the VAE-decoded full-root motion survives almost perfectly -> the VAE is not the bottleneck."""
    f = _load(os.path.join(HERE, "..", "seed_sim2sim_feasible_summary.json"))
    assert f["feasible_n"] >= 30
    assert f["decoded_survival_on_feasible"] >= 0.9, f
    assert f["frac_full_on_feasible"] >= 0.85, f
