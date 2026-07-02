"""Score the large sample for both trackers and surface issues.

Reads large/bfm/rollout_<i>.npz + large/holo/holo_<i>.npz, computes survival
(abs + ref-relative), joint error, and pelvis-depth match per category/group, and
flags problem clips (BFM failures, depth-floor under-tracking, suspicious references).
Writes large_survival.json + large_issues.json.
"""
import json
import os
import statistics as st
import numpy as np

from stage3_sim2sim.sim2sim import rollout_metrics

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "large"
man = json.load(open(f"{HERE}/{TAG}_sample.json"))


def load(p):
    d = np.load(p)
    return d["executed_qpos_36"], d["reference_qpos_36"]


rows = []
for m in man:
    bp, hp = f"{HERE}/{TAG}/bfm/rollout_{m['idx']}.npz", f"{HERE}/{TAG}/holo/holo_{m['idx']}.npz"
    if not (os.path.exists(bp) and os.path.exists(hp)):
        continue
    bex, bref = load(bp); hex_, href = load(hp)
    bm, hm = rollout_metrics(bex, bref), rollout_metrics(hex_, href)
    bn = min(len(bex), len(bref))
    rows.append({
        "idx": m["idx"], "artifact": m["artifact"], "cat": m["cat"], "group": m["group"],
        "ref_pelvis_min": round(float(bref[:bn, 2].min()), 3),
        "holo_surv": round(hm["survival"], 3), "holo_rel": round(hm["survival_rel"], 3),
        "holo_joint": round(hm["joint_rmse_deg"], 1),
        "bfm_surv": round(bm["survival"], 3), "bfm_rel": round(bm["survival_rel"], 3),
        "bfm_joint": round(bm["joint_rmse_deg"], 1),
        "bfm_depth_gap": round(float(np.abs(bex[:bn, 2] - bref[:bn, 2]).mean()), 3),
    })

print(f"scored {len(rows)}/{len(man)} clips")


def agg(rs):
    f = lambda k: round(st.mean(r[k] for r in rs), 3)
    return {"n": len(rs), "holo_surv": f("holo_surv"), "bfm_surv": f("bfm_surv"),
            "holo_rel": f("holo_rel"), "bfm_rel": f("bfm_rel"),
            "holo_joint": f("holo_joint"), "bfm_joint": f("bfm_joint"),
            "bfm_depth_gap": f("bfm_depth_gap"),
            "bfm_wins_joint": round(st.mean(r["bfm_joint"] < r["holo_joint"] for r in rs), 2)}


report = {
    "n": len(rows), "overall": agg(rows),
    "by_group": {g: agg([r for r in rows if r["group"] == g]) for g in ("standing", "near-ground")
                 if any(r["group"] == g for r in rows)},
    "by_category": {c: agg([r for r in rows if r["cat"] == c])
                    for c in sorted({r["cat"] for r in rows}) if sum(r["cat"] == c for r in rows) >= 3},
    "rows": rows,
}
json.dump(report, open(f"{HERE}/{TAG}_survival.json", "w"), indent=2)

# ---- issue flagging ----
issues = {
    "bfm_fails_holo_succeeds": [r for r in rows if r["bfm_rel"] < 0.6 and r["holo_rel"] > 0.8],
    "both_fail": [r for r in rows if r["bfm_rel"] < 0.5 and r["holo_rel"] < 0.5],
    "bfm_deep_undertrack": sorted([r for r in rows if r["ref_pelvis_min"] < 0.35 and r["bfm_depth_gap"] > 0.12],
                                  key=lambda r: -r["bfm_depth_gap"]),
    "suspicious_reference_low_pelvis": [r for r in rows if r["ref_pelvis_min"] < 0.0],
    "bfm_low_survival": sorted([r for r in rows if r["bfm_rel"] < 0.5], key=lambda r: r["bfm_rel"]),
}
json.dump({k: v[:40] for k, v in issues.items()}, open(f"{HERE}/{TAG}_issues.json", "w"), indent=2)

o = report["overall"]
print(f"\nOVERALL n={o['n']}: survival_rel HOLO {o['holo_rel']} BFM {o['bfm_rel']} | "
      f"joint HOLO {o['holo_joint']} BFM {o['bfm_joint']} | BFM wins joint {o['bfm_wins_joint']:.0%}")
for g, a in report["by_group"].items():
    print(f"  {g:11s} n={a['n']:3d}: rel HOLO {a['holo_rel']:.2f}/BFM {a['bfm_rel']:.2f}  "
          f"joint {a['holo_joint']:.0f}/{a['bfm_joint']:.0f}  bfm_depth_gap {a['bfm_depth_gap']:.3f}")
print("\nby category (survival_rel HOLO/BFM, joint HOLO/BFM):")
for c, a in report["by_category"].items():
    print(f"  {c:8s} n={a['n']:3d}: {a['holo_rel']:.2f}/{a['bfm_rel']:.2f}  {a['holo_joint']:.0f}/{a['bfm_joint']:.0f}")
print("\n=== ISSUES ===")
for k, v in issues.items():
    print(f"  {k}: {len(v)} clips")
print(f"wrote {TAG}_survival.json + {TAG}_issues.json")
