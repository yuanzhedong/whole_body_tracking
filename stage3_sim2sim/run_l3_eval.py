"""L3 end-to-end sim2sim eval: decoded-vs-original survival/tracking across clips.

For each clip: build ground-truth qpos_36, VAE-decode its features, build the hybrid
reference (decoded joints + original root), run BOTH original and decoded through the
HoloMotion tracker (MuJoCo), and report survival + tracking metrics. The decoded-vs-
original gap is the VAE's physical-executability cost (Gate D).

Run in the UniMoTok venv (needs torch + the tokenizer); the tracker is shelled out to
the OMG repo by ``run_tracker``.
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage3_sim2sim.sim2sim import (
    build_qpos36_from_artifact, build_hybrid_qpos36, run_tracker, rollout_metrics,
)
from stage3_sim2sim.decode_to_qpos36 import qpos36_to_features
from stage3_sim2sim.vae_decode_clip import load_vae, decode_features

ONNX = ("/scratch/user/yzdong/OMG-models/holomotion_dl/"
        "HoloMotion_motion_tracking_model_v1.3.1/exported/motion_tracking_model.onnx")
NORM = "/scratch/user/yzdong/OMG-Data/umt/g1_seed_full_yup/normalization.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--umt-root", default="UniMoTok")
    ap.add_argument("--omg-root", default="/ws/user/yzdong/src/github/OMG")
    ap.add_argument("--clips", nargs="+", required=True, help="artifact motion.npz paths")
    ap.add_argument("--out", default="/tmp/sim2sim_l3")
    ap.add_argument("--num-frames", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    norm = np.load(NORM)
    model, ws = load_vae(args.cfg, args.ckpt, args.umt_root, device=args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in args.clips:
        name = Path(clip).parent.name[:36]
        qpos_gt = build_qpos36_from_artifact(clip)
        if qpos_gt.shape[0] < ws:
            continue
        feats = qpos36_to_features(qpos_gt, 1 / 30)
        rec = decode_features(model, feats[:ws], norm["mean"], norm["std"],
                              device=args.device, window=ws)
        hybrid = build_hybrid_qpos36(rec, qpos_gt[:ws])
        jdeg = float(np.sqrt(np.mean((rec[:, 12:41] - feats[:ws, 12:41]) ** 2)) * 180 / np.pi)
        res = {"clip": name, "decode_joint_rmse_deg": round(jdeg, 2)}
        for tag, q in [("orig", qpos_gt[:ws]), ("decoded", hybrid)]:
            roll = run_tracker(q, fps=30, out_dir=out / name / tag, onnx_path=ONNX,
                               omg_root=args.omg_root, num_frames=args.num_frames)
            d = np.load(roll)
            m = rollout_metrics(d["executed_qpos_36"], d["reference_qpos_36"])
            res[f"{tag}_survival"] = round(m["survival"], 3)
            res[f"{tag}_track_deg"] = round(m["joint_rmse_deg"], 1)
        rows.append(res)
        print(f"{name:36s} decode={res['decode_joint_rmse_deg']:5.2f}deg | "
              f"surv orig={res['orig_survival']:.2f} dec={res['decoded_survival']:.2f} | "
              f"track orig={res['orig_track_deg']:.1f} dec={res['decoded_track_deg']:.1f}", flush=True)

    (out / "l3_results.json").write_text(json.dumps(rows, indent=2))
    surv_o = np.mean([r["orig_survival"] for r in rows])
    surv_d = np.mean([r["decoded_survival"] for r in rows])
    print(f"\n=== Gate D ({len(rows)} clips) === survival orig={surv_o:.3f} decoded={surv_d:.3f}  "
          f"mean decode joint RMSE={np.mean([r['decode_joint_rmse_deg'] for r in rows]):.2f}deg")
    print(f"results -> {out/'l3_results.json'}")


if __name__ == "__main__":
    main()
