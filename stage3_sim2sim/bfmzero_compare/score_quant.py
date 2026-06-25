"""Aggregate quantitative analysis over all grounded near-ground clips.

Reads quant/bfm_<idx>.npz + quant/holo_<idx>.npz, scores both with the shared
rollout_metrics, and writes quant_analysis.json (per-clip rows + per-category and
overall aggregates). Run in any env with numpy.
"""
import json
import os
import statistics as st
import numpy as np

from stage3_sim2sim.sim2sim import rollout_metrics
from stage3_sim2sim.bfmzero_compare.quant_clips import QUANT_CLIPS

DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant")


def metrics(npz):
    d = np.load(npz)
    return rollout_metrics(d["executed_qpos_36"], d["reference_qpos_36"])


rows = []
for idx, art, cat, label in QUANT_CLIPS:
    b, h = f"{DST}/bfm_{idx}.npz", f"{DST}/holo_{idx}.npz"
    if not (os.path.exists(b) and os.path.exists(h)):
        print(f"[{idx}] missing rollout(s), skip"); continue
    bm, hm = metrics(b), metrics(h)
    rows.append({
        "idx": idx, "clip": art.replace(":v0", ""), "cat": cat, "label": label,
        "ref_z_min": round(hm["ref_z_min"], 3),
        "holo_surv": round(hm["survival"], 3), "holo_rel": round(hm["survival_rel"], 3),
        "holo_joint": round(hm["joint_rmse_deg"], 1), "holo_zmae": round(hm["root_z_mae"], 3),
        "bfm_surv": round(bm["survival"], 3), "bfm_rel": round(bm["survival_rel"], 3),
        "bfm_joint": round(bm["joint_rmse_deg"], 1), "bfm_zmae": round(bm["root_z_mae"], 3),
    })


def agg(rs):
    mean = lambda k: round(st.mean(r[k] for r in rs), 3)
    med = lambda k: round(st.median(r[k] for r in rs), 3)
    return {
        "n": len(rs),
        "holo_surv_mean": mean("holo_surv"), "bfm_surv_mean": mean("bfm_surv"),
        "holo_rel_mean": mean("holo_rel"), "bfm_rel_mean": mean("bfm_rel"),
        "holo_joint_mean": mean("holo_joint"), "bfm_joint_mean": mean("bfm_joint"),
        "holo_joint_med": med("holo_joint"), "bfm_joint_med": med("bfm_joint"),
        "bfm_wins_joint": sum(r["bfm_joint"] < r["holo_joint"] for r in rs),
        "bfm_rel_ge_0.9": sum(r["bfm_rel"] >= 0.9 for r in rs),
        "holo_rel_ge_0.9": sum(r["holo_rel"] >= 0.9 for r in rs),
    }


cats = sorted({r["cat"] for r in rows})
report = {
    "n_clips": len(rows),
    "overall": agg(rows),
    "by_category": {c: agg([r for r in rows if r["cat"] == c]) for c in cats},
    "rows": rows,
}
json.dump(report, open(f"{DST}/../quant_analysis.json", "w"), indent=2)

print(f"=== Quantitative analysis: {len(rows)} grounded near-ground clips ===\n")
print(f"{'clip':46s} {'cat':6s} {'refz':>5s} | {'HOLO surv/rel/jt':>16s} | {'BFM surv/rel/jt':>16s}")
for r in rows:
    print(f"{r['label'][:46]:46s} {r['cat']:6s} {r['ref_z_min']:5.2f} | "
          f"{r['holo_surv']:.2f}/{r['holo_rel']:.2f}/{r['holo_joint']:5.1f}  | "
          f"{r['bfm_surv']:.2f}/{r['bfm_rel']:.2f}/{r['bfm_joint']:5.1f}")
o = report["overall"]
print(f"\nOVERALL (n={o['n']}): survival_rel  HOLO {o['holo_rel_mean']:.2f}  BFM {o['bfm_rel_mean']:.2f}  | "
      f"joint°  HOLO {o['holo_joint_mean']:.1f}  BFM {o['bfm_joint_mean']:.1f}  | "
      f"BFM lower joint on {o['bfm_wins_joint']}/{o['n']}")
for c, a in report["by_category"].items():
    print(f"  {c:6s} n={a['n']}: rel HOLO {a['holo_rel_mean']:.2f}/BFM {a['bfm_rel_mean']:.2f}  "
          f"joint HOLO {a['holo_joint_mean']:.1f}/BFM {a['bfm_joint_mean']:.1f}")
print(f"\nwrote quant_analysis.json")
