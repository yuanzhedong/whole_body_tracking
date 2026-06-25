"""Re-score BOTH trackers on the 4 near-ground clips with the same rollout_metrics.

Reads the BFM-Zero rollouts (bfmzero_compare/rollout_clip*.npz) and the HoloMotion
rollouts (/tmp/holo_clip*/.../holomotion_rollout.npz), writes a unified comparison
JSON with absolute + reference-relative survival, joint error and pelvis z-MAE.
"""
import glob
import json
import os
import numpy as np

from stage3_sim2sim.sim2sim import rollout_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = [
    ("clip0", "crouch_idle_right_R_003__A247_M", "deep crouch (idle)"),
    ("clip1", "crouch_ff_start_270_R_001__A197_M", "crouch + turn"),
    ("clip2", "sit_on_chair_stop_R_001__A047", "sit down"),
    ("clip3", "squat_001__A360", "squat"),
]


def holo_rollout(cid):
    hits = glob.glob(f"/tmp/holo_{cid}/**/holomotion_rollout.npz", recursive=True)
    return hits[0] if hits else None


rows = []
for cid, name, desc in CLIPS:
    b = np.load(f"{HERE}/rollout_{cid}.npz")
    bm = rollout_metrics(b["executed_qpos_36"], b["reference_qpos_36"])
    hp = holo_rollout(cid)
    h = np.load(hp)
    hm = rollout_metrics(h["executed_qpos_36"], h["reference_qpos_36"])
    rows.append({
        "clip": name, "cid": cid, "motion": desc,
        "holo_survival": round(hm["survival"], 2), "holo_survival_rel": round(hm["survival_rel"], 2),
        "holo_joint_deg": round(hm["joint_rmse_deg"], 1), "holo_z_mae": round(hm["root_z_mae"], 3),
        "bfm_survival": round(bm["survival"], 2), "bfm_survival_rel": round(bm["survival_rel"], 2),
        "bfm_joint_deg": round(bm["joint_rmse_deg"], 1), "bfm_z_mae": round(bm["root_z_mae"], 3),
        "ref_z_min": round(bm["ref_z_min"], 2), "n": int(bm["n_frames"]),
    })

json.dump(rows, open(f"{HERE}/bfmzero_vs_holomotion.json", "w"), indent=2)
hdr = f"{'clip':20s} {'refzmin':>7s} | {'HOLO surv/rel/jt°':>18s} | {'BFM surv/rel/jt°':>18s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['motion']:20s} {r['ref_z_min']:7.2f} | "
          f"{r['holo_survival']:5.2f}/{r['holo_survival_rel']:4.2f}/{r['holo_joint_deg']:5.1f}   | "
          f"{r['bfm_survival']:5.2f}/{r['bfm_survival_rel']:4.2f}/{r['bfm_joint_deg']:5.1f}")
print(f"\nwrote {HERE}/bfmzero_vs_holomotion.json")
