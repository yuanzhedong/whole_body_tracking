"""Comprehensive reconstruction-quality report for a G1 VAE on the seed dataset.

Evaluates a checkpoint on a large held-out val sample and reports, overall and broken
down by motion category and by dynamism (static vs dynamic):
  - joint RMSE (degrees)        -- the headline body-pose error
  - per-feature-block normalized MSE (rot6d / lin_vel / ang_vel / joints)
  - root orientation GEODESIC error (degrees)  -- the rotation-correct metric
Writes a JSON summary. Run in the UniMoTok env.
"""
from __future__ import annotations
import argparse, glob, json, os, random, sys
import numpy as np

sys.path.insert(0, ".")


def categorize(name):
    n = name.lower()
    for c in ("walk", "run", "jog", "jump", "dance", "kick", "turn", "squat", "punch",
              "wave", "bow", "crouch", "sit", "step", "reach", "idle", "throw", "sprint", "spin"):
        if c in n:
            return c
    return "other"


def geodesic_deg(Ra, Rb):
    rel = np.einsum("tij,tkj->tik", Ra, Rb)
    tr = np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def main():
    import torch
    from omegaconf import OmegaConf
    from multimodal_tokenizers.models.build_model import build_model
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "UniMoTok"))
    from stage3_sim2sim.rotation_utils import rot6d_to_matrix

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/seed_recon_report.json")
    a = ap.parse_args()

    cfg = OmegaConf.load(a.cfg); ws = int(cfg.DATASET.window_size)
    data_dir = str(cfg.DATASET.data_dir)
    model = build_model(cfg)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=False); model = model.eval().to(a.device)
    nz = np.load(f"{data_dir}/normalization.npz")
    mean, std = nz["mean"].astype(np.float32), np.clip(nz["std"].astype(np.float32), 1e-6, None)
    vals = sorted(glob.glob(f"{data_dir}/val/*.npz")); random.seed(0); random.shuffle(vals)

    rows = []
    for vp in vals:
        m = np.load(vp)["motion"].astype(np.float32)
        if len(m) < ws:
            continue
        raw = m[:ws]; w = (raw - mean) / std
        with torch.no_grad():
            rec = model.vae(torch.tensor(w)[None].to(a.device))["rec_pose"].cpu().numpy()[0][:ws]
        rec_raw = rec * std + mean
        jt = raw[:, 12:41]; jt_r = rec_raw[:, 12:41]
        nm = lambda i, j: float(np.mean((rec[:, i:j] - w[:, i:j]) ** 2))
        rows.append({
            "cat": categorize(os.path.basename(vp)),
            "motion": float(jt.std(0).mean()),
            "joint_rmse_deg": float(np.sqrt(np.mean((jt - jt_r) ** 2)) * 180 / np.pi),
            "nm_rot6d": nm(0, 6), "nm_linvel": nm(6, 9), "nm_angvel": nm(9, 12), "nm_joints": nm(12, 41),
            "root_geo_deg": float(geodesic_deg(rot6d_to_matrix(rec_raw[:, 0:6]),
                                               rot6d_to_matrix(raw[:, 0:6])).mean()),
        })
        if len(rows) >= a.n:
            break

    def agg(rs):
        return {"n": len(rs),
                "joint_rmse_deg": round(float(np.mean([r["joint_rmse_deg"] for r in rs])), 2),
                "root_geo_deg": round(float(np.mean([r["root_geo_deg"] for r in rs])), 2),
                "nm_joints": round(float(np.mean([r["nm_joints"] for r in rs])), 3),
                "nm_rot6d": round(float(np.mean([r["nm_rot6d"] for r in rs])), 3)}

    med = np.median([r["motion"] for r in rows])
    report = {
        "ckpt": a.ckpt, "n_clips": len(rows),
        "overall": agg(rows),
        "static": agg([r for r in rows if r["motion"] < med]),
        "dynamic": agg([r for r in rows if r["motion"] >= med]),
        "by_category": {c: agg([r for r in rows if r["cat"] == c])
                        for c in sorted({r["cat"] for r in rows})
                        if sum(r["cat"] == c for r in rows) >= 3},
    }
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"=== seed reconstruction report ({len(rows)} val clips) ===")
    print(f"OVERALL : joint {report['overall']['joint_rmse_deg']}deg  root_geo {report['overall']['root_geo_deg']}deg  "
          f"nm_joints {report['overall']['nm_joints']}")
    print(f"STATIC  : joint {report['static']['joint_rmse_deg']}deg  root_geo {report['static']['root_geo_deg']}deg")
    print(f"DYNAMIC : joint {report['dynamic']['joint_rmse_deg']}deg  root_geo {report['dynamic']['root_geo_deg']}deg")
    print("by category (joint_rmse_deg / root_geo_deg):")
    for c, v in report["by_category"].items():
        print(f"  {c:8s} n={v['n']:3d}  {v['joint_rmse_deg']:5.2f} / {v['root_geo_deg']:5.2f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
