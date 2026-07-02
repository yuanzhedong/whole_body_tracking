"""Score VAE-decoded vs original motion, both executed through BFM-Zero.

Answers the sim2sim question for the BONES-SEED -> UniMoTok VAE -> BFM-Zero
pipeline: does VAE-decoded motion stay physically executable? Compares, per clip,
the original-through-BFM rollout (quant/bfm_<i>.npz) to the decoded-through-BFM
rollout (quant/dec_<i>.npz), each scored against its own reference. Writes
decoded_analysis.json.
"""
import json
import os
import statistics as st
import numpy as np

from stage3_sim2sim.sim2sim import rollout_metrics
from stage3_sim2sim.bfmzero_compare.quant_clips import QUANT_CLIPS

DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant")


def m(npz):
    d = np.load(npz)
    return rollout_metrics(d["executed_qpos_36"], d["reference_qpos_36"])


rows = []
for idx, art, cat, label in QUANT_CLIPS:
    o, d = f"{DST}/bfm_{idx}.npz", f"{DST}/dec_{idx}.npz"
    if not (os.path.exists(o) and os.path.exists(d)):
        continue
    om, dm = m(o), m(d)
    rows.append({
        "idx": idx, "clip": art.replace(":v0", ""), "cat": cat, "label": label,
        "orig_surv_rel": round(om["survival_rel"], 3), "dec_surv_rel": round(dm["survival_rel"], 3),
        "orig_joint": round(om["joint_rmse_deg"], 1), "dec_joint": round(dm["joint_rmse_deg"], 1),
        "orig_surv": round(om["survival"], 3), "dec_surv": round(dm["survival"], 3),
    })


def mean(rs, k):
    return round(st.mean(r[k] for r in rs), 3)


overall = {
    "n": len(rows),
    "orig_surv_rel_mean": mean(rows, "orig_surv_rel"), "dec_surv_rel_mean": mean(rows, "dec_surv_rel"),
    "orig_surv_mean": mean(rows, "orig_surv"), "dec_surv_mean": mean(rows, "dec_surv"),
    "orig_joint_mean": mean(rows, "orig_joint"), "dec_joint_mean": mean(rows, "dec_joint"),
    "dec_rel_ge_0.9": sum(r["dec_surv_rel"] >= 0.9 for r in rows),
}
cats = sorted({r["cat"] for r in rows})
report = {"n_clips": len(rows), "overall": overall,
          "by_category": {c: {"n": sum(r["cat"] == c for r in rows),
                              "orig_surv_rel": mean([r for r in rows if r["cat"] == c], "orig_surv_rel"),
                              "dec_surv_rel": mean([r for r in rows if r["cat"] == c], "dec_surv_rel")}
                          for c in cats},
          "rows": rows}
json.dump(report, open(f"{DST}/../decoded_analysis.json", "w"), indent=2)

print(f"=== VAE-decoded vs original, both through BFM-Zero ({len(rows)} clips) ===\n")
print(f"{'clip':40s} {'cat':6s} | {'orig rel/jt':>12s} | {'decoded rel/jt':>14s}")
for r in rows:
    print(f"{r['label'][:40]:40s} {r['cat']:6s} | {r['orig_surv_rel']:.2f}/{r['orig_joint']:5.1f}  | "
          f"{r['dec_surv_rel']:.2f}/{r['dec_joint']:5.1f}")
o = overall
print(f"\nOVERALL (n={o['n']}): survival_rel orig {o['orig_surv_rel_mean']:.2f} -> decoded {o['dec_surv_rel_mean']:.2f}  "
      f"| joint° orig {o['orig_joint_mean']:.1f} -> decoded {o['dec_joint_mean']:.1f}  "
      f"| decoded rel>=0.9 on {o['dec_rel_ge_0.9']}/{o['n']}")
print("wrote decoded_analysis.json")
