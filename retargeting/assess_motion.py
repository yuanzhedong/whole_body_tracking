"""Quality assessment for a retargeted motion (the npz produced by scripts/csv_to_npz.py).

Answers the practical question "is this retargeted motion good enough to train on, or do I need
better retargeting (GMR)?" by measuring common retargeting artifacts from the FK'd body poses:
  - ground penetration : lowest body z over the clip (should be >~ 0; very negative = feet sink)
  - foot skate         : mean horizontal foot speed while the foot is planted (high = sliding)
  - jitter             : 99th-pct joint acceleration (high = noisy/unstable retarget)
  - joint-limit viol.  : fraction of frames any joint is outside the G1 limits
  - root sanity        : pelvis height range, total travel, max per-frame jump

Compares against an optional baseline npz (e.g. the LAFAN1 walk) so the numbers have a reference,
and prints a pass/fail verdict against thresholds.

Usage:
  python retargeting/assess_motion.py --npz /tmp/motion.npz [--baseline /tmp/wbt_fix/walk.npz] \
      [--out retargeting/out/assess.json]
"""
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_bodies import G1_BODY_NAMES, PELVIS_IDX, FOOT_IDX  # noqa: E402
from hf_to_csv import JOINT_LIMITS  # noqa: E402

FOOT_CONTACT_Z = 0.10  # ankle_roll frame below this => foot considered planted (frame ~0.035 when down)

# pass thresholds (tuned to LAFAN1-walk being comfortably inside)
THRESHOLDS = {
    "ground_penetration_m": -0.05,   # min body z must be >= this (small clip ok)
    "foot_skate_mps": 0.15,          # planted-foot horizontal speed
    "joint_jitter_p99": 80.0,        # rad/s^2
    "joint_limit_viol_frac": 0.02,
}


def assess(npz_path: str) -> dict:
    d = np.load(npz_path)
    bp = d["body_pos_w"].astype(np.float64)        # [T, nb, 3]
    jp = d["joint_pos"].astype(np.float64)         # [T, 29]
    fps = float(np.atleast_1d(d["fps"])[0])
    T, nb = bp.shape[0], bp.shape[1]
    assert nb == len(G1_BODY_NAMES), f"expected {len(G1_BODY_NAMES)} bodies, got {nb}"
    dt = 1.0 / fps

    # ground penetration: lowest point of the whole robot over time
    ground_penetration = float(bp[:, :, 2].min())

    # foot skate: horizontal foot speed while planted
    skate_vals = []
    for fi in FOOT_IDX:
        z = bp[:, fi, 2]
        xy = bp[:, fi, :2]
        v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt  # [T-1]
        planted = z[:-1] < FOOT_CONTACT_Z
        if planted.any():
            skate_vals.append(float(v[planted].mean()))
    foot_skate = float(np.mean(skate_vals)) if skate_vals else float("nan")

    # jitter: joint acceleration 99th percentile
    if T >= 3:
        acc = np.diff(jp, n=2, axis=0) / (dt * dt)
        jitter_p99 = float(np.percentile(np.abs(acc), 99))
    else:
        jitter_p99 = float("nan")

    # joint-limit violations — fraction of frames AND how far over (magnitude matters:
    # the proven-good LAFAN1 walk also nicks limits in most frames, but only barely).
    lows = np.array([lo for _, lo, _ in JOINT_LIMITS])
    highs = np.array([hi for _, _, hi in JOINT_LIMITS])
    over = np.maximum(jp - highs[None, :], 0.0) + np.maximum(lows[None, :] - jp, 0.0)  # rad outside
    viol = over > 0
    jl_frac = float(viol.any(axis=1).mean())
    jl_excess_max = float(over.max())
    jl_excess_mean = float(over[viol].mean()) if viol.any() else 0.0

    # root sanity
    pelvis = bp[:, PELVIS_IDX, :]
    root_jump = float(np.linalg.norm(np.diff(pelvis, axis=0), axis=1).max()) if T > 1 else 0.0

    return {
        "npz": npz_path,
        "frames": int(T),
        "fps": fps,
        "duration_s": round(T / fps, 2),
        "ground_penetration_m": round(ground_penetration, 4),
        "foot_skate_mps": round(foot_skate, 4),
        "joint_jitter_p99_rad_s2": round(jitter_p99, 2),
        "joint_limit_viol_frac": round(jl_frac, 4),
        "joint_limit_excess_max_rad": round(jl_excess_max, 4),
        "joint_limit_excess_mean_rad": round(jl_excess_mean, 4),
        "pelvis_z_min": round(float(pelvis[:, 2].min()), 3),
        "pelvis_z_max": round(float(pelvis[:, 2].max()), 3),
        "root_travel_m": round(float(np.linalg.norm(pelvis[-1, :2] - pelvis[0, :2])), 3),
        "root_max_frame_jump_m": round(root_jump, 4),
    }


def verdict(rep: dict, base: dict | None = None) -> dict:
    """Prefer a baseline-relative verdict: the LAFAN1 walk is known to train to ~99% success,
    so 'comparable to baseline' is the real bar. Absolute checks are a fallback heuristic."""
    if base is not None:
        def le(metric, factor, floor=0.0):
            return rep[metric] <= max(base[metric] * factor, floor)
        checks = {
            # penetration: not meaningfully deeper than baseline (allow 5cm slack)
            "ground_penetration": rep["ground_penetration_m"] >= base["ground_penetration_m"] - 0.05,
            # foot skate within 1.5x baseline
            "foot_skate": le("foot_skate_mps", 1.5),
            # jitter within 3x baseline (fast clips are naturally jerkier)
            "jitter": le("joint_jitter_p99_rad_s2", 3.0),
            # joint-limit OVERSHOOT magnitude within 2x baseline (fraction is uninformative)
            "joint_limit_excess": le("joint_limit_excess_max_rad", 2.0, floor=0.10),
            "no_teleport": rep["root_max_frame_jump_m"] <= max(base["root_max_frame_jump_m"] * 3, 0.20),
        }
        mode = "baseline-relative"
    else:
        checks = {
            "ground_penetration": rep["ground_penetration_m"] >= THRESHOLDS["ground_penetration_m"],
            "foot_skate": not np.isnan(rep["foot_skate_mps"]) and rep["foot_skate_mps"] <= THRESHOLDS["foot_skate_mps"],
            "jitter": not np.isnan(rep["joint_jitter_p99_rad_s2"]) and rep["joint_jitter_p99_rad_s2"] <= THRESHOLDS["joint_jitter_p99"],
            "joint_limit_excess": rep["joint_limit_excess_max_rad"] <= 0.20,
            "no_teleport": rep["root_max_frame_jump_m"] <= 0.20,
        }
        mode = "absolute-heuristic"
    passed = all(checks.values())
    return {
        "mode": mode,
        "checks": checks,
        "passed": passed,
        "recommendation": ("quality comparable to known-good LAFAN1 -> usable, GMR optional"
                           if passed else
                           "some metric notably worse than baseline -> inspect viz / train smoke before trusting"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--baseline", help="reference npz (e.g. LAFAN1 walk) for comparison")
    p.add_argument("--out", default="retargeting/out/assess.json")
    args = p.parse_args()

    rep = assess(args.npz)
    base = assess(args.baseline) if (args.baseline and os.path.isfile(args.baseline)) else None
    rep["verdict"] = verdict(rep, base)
    out = {"motion": rep}
    if base is not None:
        out["baseline"] = base
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
