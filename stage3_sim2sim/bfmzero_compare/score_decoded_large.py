"""Score VAE-decoded vs original motion through BFM-Zero at scale.

For each decoded clip (large/dec/rollout_<decidx>.npz) compares to the original-motion
rollout (large/bfm/rollout_<largeidx>.npz, via large_decoded_map.json), each scored
against its own reference. Confirms the VAE latent preserves physical executability
across the distribution. Writes decoded_large.json.
"""
import json
import os
import statistics as st
import numpy as np

from stage3_sim2sim.sim2sim import rollout_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
mapping = json.load(open(f"{HERE}/large_decoded_map.json"))
man = {m["idx"]: m for m in json.load(open(f"{HERE}/large_sample.json"))}


def m(p):
    d = np.load(p)
    return rollout_metrics(d["executed_qpos_36"], d["reference_qpos_36"])


rows = []
for decidx, largeidx in enumerate(mapping):
    dp = f"{HERE}/large/dec/rollout_{decidx}.npz"
    op = f"{HERE}/large/bfm/rollout_{largeidx}.npz"
    if not (os.path.exists(dp) and os.path.exists(op)):
        continue
    dm, om = m(dp), m(op)
    rows.append({"cat": man[largeidx]["cat"], "group": man[largeidx]["group"],
                 "orig_rel": om["survival_rel"], "dec_rel": dm["survival_rel"],
                 "orig_joint": om["joint_rmse_deg"], "dec_joint": dm["joint_rmse_deg"]})


def mean(rs, k):
    return round(st.mean(r[k] for r in rs), 3)


overall = {"n": len(rows), "orig_rel": mean(rows, "orig_rel"), "dec_rel": mean(rows, "dec_rel"),
           "orig_joint": mean(rows, "orig_joint"), "dec_joint": mean(rows, "dec_joint"),
           "dec_rel_ge_0.9_frac": round(st.mean(r["dec_rel"] >= 0.9 for r in rows), 3),
           "dec_within_5pct_of_orig": round(st.mean(r["dec_rel"] >= 0.95 * r["orig_rel"] for r in rows), 3)}
report = {"overall": overall,
          "by_group": {g: {"n": sum(r["group"] == g for r in rows),
                           "orig_rel": mean([r for r in rows if r["group"] == g], "orig_rel"),
                           "dec_rel": mean([r for r in rows if r["group"] == g], "dec_rel")}
                       for g in ("standing", "near-ground") if any(r["group"] == g for r in rows)}}
json.dump(report, open(f"{HERE}/decoded_large.json", "w"), indent=2)
o = overall
print(f"=== VAE-decoded vs original through BFM-Zero, scaled ({o['n']} clips) ===")
print(f"survival_rel  orig {o['orig_rel']:.3f} -> decoded {o['dec_rel']:.3f}  | "
      f"joint  orig {o['orig_joint']:.1f} -> decoded {o['dec_joint']:.1f}")
print(f"decoded survival_rel >= 0.9 on {o['dec_rel_ge_0.9_frac']:.0%}; "
      f"within 5% of original on {o['dec_within_5pct_of_orig']:.0%}")
for g, a in report["by_group"].items():
    print(f"  {g:11s} n={a['n']:3d}: orig {a['orig_rel']:.2f} -> decoded {a['dec_rel']:.2f}")
print("wrote decoded_large.json")
