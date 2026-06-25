"""VAE-decode robustness check over many clips -> finds pipeline issues.

For each clip: original qpos -> 41-D features -> VAE encode/decode -> full-root
qpos_36. Flags NaN/Inf, out-of-range joints, large root drift, and reconstruction
outliers. Writes decode_robustness.json. Run in the OMG env (torch + UniMoTok).
"""
import json
import os
import sys
import numpy as np

from stage3_sim2sim.sim2sim import build_qpos36_from_artifact
from stage3_sim2sim.decode_to_qpos36 import qpos36_to_features, features_to_qpos36
from stage3_sim2sim.vae_decode_clip import load_vae, decode_features
from stage3_sim2sim.bfmzero_compare.build_large_sample import ARTROOT

UMT = "UniMoTok"
CFG = f"{UMT}/configs/config_g1_seed_512_fixed.yaml"
CKPT = f"{UMT}/experiments/_compare/g1_seed_512_fixed_FINAL.ckpt"
HERE = os.path.dirname(os.path.abspath(__file__))


def main(n=200):
    from omegaconf import OmegaConf
    model, ws = load_vae(CFG, CKPT, UMT, device="cpu")
    nz = np.load(f"{str(OmegaConf.load(CFG).DATASET.data_dir)}/normalization.npz")
    mean, std = nz["mean"], nz["std"]
    man = json.load(open(f"{HERE}/large_sample.json"))[:n]

    issues = {"nan_inf": [], "joint_out_of_range": [], "root_drift": [], "recon_outlier": [], "too_short": []}
    ok = 0
    for m in man:
        art = m["artifact"]
        q = build_qpos36_from_artifact(f"{ARTROOT}/{art}/motion.npz")
        if q.shape[0] < ws:
            issues["too_short"].append(art); continue
        feats = qpos36_to_features(q, 1 / 30)
        rec = decode_features(model, feats, mean, std, device="cpu")
        dec = features_to_qpos36(rec, 1 / 30, root_pos0_zup=q[0, :3])
        if not np.isfinite(rec).all() or not np.isfinite(dec).all():
            issues["nan_inf"].append(art); continue
        jt = np.degrees(np.abs(dec[:, 7:36]))
        if jt.max() > 220:
            issues["joint_out_of_range"].append({"clip": art, "max_deg": round(float(jt.max()), 0)})
        # root drift: decoded full-root xy vs original (the known double-yup risk)
        drift = float(np.linalg.norm(dec[:, :2] - q[:len(dec), :2], axis=1).max())
        if drift > 0.5:
            issues["root_drift"].append({"clip": art, "drift_m": round(drift, 2)})
        # recon quality
        rmse = float(np.sqrt(((np.degrees(q[:len(rec), 7:36]) - np.degrees(rec[:, 12:41])) ** 2).mean()))
        if rmse > 25:
            issues["recon_outlier"].append({"clip": art, "joint_rmse_deg": round(rmse, 1)})
        ok += 1
    summary = {"n_checked": len(man), "n_ok": ok, **{k: len(v) for k, v in issues.items()},
               "details": {k: v[:25] for k, v in issues.items()}}
    json.dump(summary, open(f"{HERE}/decode_robustness.json", "w"), indent=2)
    print(json.dumps({k: summary[k] for k in summary if k != "details"}, indent=2))
    print("wrote decode_robustness.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
