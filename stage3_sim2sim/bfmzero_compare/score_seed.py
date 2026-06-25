"""Seed-dataset survival rate: HoloMotion vs BFM-Zero on the 40-clip sample.

Scores seed/bfm_<i>.npz and seed/holo_<i>.npz with the shared rollout_metrics and
aggregates overall, by category, and by standing-vs-near-ground split. Writes
seed_survival.json.
"""
import json
import os
import statistics as st
import numpy as np

from stage3_sim2sim.sim2sim import rollout_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
NEAR = ("crouch", "squat", "sit", "kneel", "crawl", "stoop")


def cat(n):
    n = n.lower()
    # near-ground keywords first so compound names (e.g. idle_crawl_stop) classify correctly
    for c in ("crouch", "squat", "crawl", "kneel", "stoop", "sit",
              "walk", "run", "jog", "sprint", "jump", "dance", "kick", "punch",
              "turn", "step", "wave", "bow", "throw", "reach", "idle", "spin", "stand"):
        if c in n:
            return c
    return "other"


def m(npz):
    d = np.load(npz)
    return rollout_metrics(d["executed_qpos_36"], d["reference_qpos_36"])


manifest = json.load(open(f"{HERE}/seed_sample.json"))
rows = []
for e in manifest:
    i = e["idx"]
    b, h = f"{HERE}/seed/bfm_{i}.npz", f"{HERE}/seed/holo_{i}.npz"
    if not (os.path.exists(b) and os.path.exists(h)):
        continue
    bm, hm = m(b), m(h)
    c = cat(e["artifact"])
    rows.append({
        "idx": i, "artifact": e["artifact"], "cat": c,
        "group": "near-ground" if c in NEAR else "standing",
        "holo_surv": round(hm["survival"], 3), "holo_rel": round(hm["survival_rel"], 3),
        "bfm_surv": round(bm["survival"], 3), "bfm_rel": round(bm["survival_rel"], 3),
        "holo_joint": round(hm["joint_rmse_deg"], 1), "bfm_joint": round(bm["joint_rmse_deg"], 1),
    })


def agg(rs):
    mean = lambda k: round(st.mean(r[k] for r in rs), 3)
    return {"n": len(rs), "holo_surv": mean("holo_surv"), "bfm_surv": mean("bfm_surv"),
            "holo_rel": mean("holo_rel"), "bfm_rel": mean("bfm_rel"),
            "holo_joint": mean("holo_joint"), "bfm_joint": mean("bfm_joint")}


report = {
    "n": len(rows),
    "overall": agg(rows),
    "by_group": {g: agg([r for r in rows if r["group"] == g]) for g in ("standing", "near-ground")
                 if any(r["group"] == g for r in rows)},
    "by_category": {c: agg([r for r in rows if r["cat"] == c])
                    for c in sorted({r["cat"] for r in rows})
                    if sum(r["cat"] == c for r in rows) >= 2},
    "rows": rows,
}
json.dump(report, open(f"{HERE}/seed_survival.json", "w"), indent=2)

o = report["overall"]
print(f"=== Seed-dataset survival ({len(rows)} clips) ===\n")
print(f"OVERALL survival  abs: HOLO {o['holo_surv']:.2f}  BFM {o['bfm_surv']:.2f}   |  "
      f"ref-rel: HOLO {o['holo_rel']:.2f}  BFM {o['bfm_rel']:.2f}   |  "
      f"joint° HOLO {o['holo_joint']:.1f}  BFM {o['bfm_joint']:.1f}")
for g, a in report["by_group"].items():
    print(f"  {g:11s} n={a['n']:2d}: abs HOLO {a['holo_surv']:.2f}/BFM {a['bfm_surv']:.2f}  "
          f"rel HOLO {a['holo_rel']:.2f}/BFM {a['bfm_rel']:.2f}")
print("\nby category:")
for c, a in report["by_category"].items():
    print(f"  {c:8s} n={a['n']:2d}: abs HOLO {a['holo_surv']:.2f}/BFM {a['bfm_surv']:.2f}")
print("\nwrote seed_survival.json")
