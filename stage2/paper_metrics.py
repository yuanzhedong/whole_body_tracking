"""Report UniMoTok-paper-style reconstruction metrics for a G1 VAE on our clips.
Paper biomech headline = joint-angle RMSE (rad), target < 0.10. Also per-joint-GROUP RMSE
(legs/arms/waist, using the CORRECT interleaved G1 joint order) + root orient error.
No Isaac — pure decode + compare. Run:
  .venv/bin/python stage2/paper_metrics.py --vae_ckpt <ckpt> --dataset_dir stage2/out/g1_dataset_gated8
"""
import argparse, os, sys, json, types
import numpy as np
sys.path.insert(0, "/ws/user/yzdong/src/github/whole_body_tracking/UniMoTok")
import torch


def load_vae(ckpt_path):
    from multimodal_tokenizers.archs.mld_vae import MldVaeBiomechanics
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ta = c["hyper_parameters"]["tokenizer_arch"]
    vae = MldVaeBiomechanics(types.SimpleNamespace(**ta["params"]))
    sd = {k[4:]: v for k, v in c["state_dict"].items() if k.startswith("vae.")}
    vae.load_state_dict(sd); vae.eval()
    return vae


def load_norm(dataset_dir):
    p = np.load(os.path.join(dataset_dir, "normalization.npz"))
    return p["mean"].astype(np.float32), np.maximum(p["std"].astype(np.float32), 1e-6)

NAMES = ["left_hip_pitch_joint","right_hip_pitch_joint","waist_yaw_joint","left_hip_roll_joint",
    "right_hip_roll_joint","waist_roll_joint","left_hip_yaw_joint","right_hip_yaw_joint",
    "waist_pitch_joint","left_knee_joint","right_knee_joint","left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint","left_ankle_pitch_joint","right_ankle_pitch_joint",
    "left_shoulder_roll_joint","right_shoulder_roll_joint","left_ankle_roll_joint",
    "right_ankle_roll_joint","left_shoulder_yaw_joint","right_shoulder_yaw_joint","left_elbow_joint",
    "right_elbow_joint","left_wrist_roll_joint","right_wrist_roll_joint","left_wrist_pitch_joint",
    "right_wrist_pitch_joint","left_wrist_yaw_joint","right_wrist_yaw_joint"]
LEG = [i for i, n in enumerate(NAMES) if any(s in n for s in ("hip", "knee", "ankle"))]
ARM = [i for i, n in enumerate(NAMES) if any(s in n for s in ("shoulder", "elbow", "wrist"))]
WAIST = [i for i, n in enumerate(NAMES) if "waist" in n]


def reconstruct(vae, feat, mean, std, window=128, stride=64):
    feat_n = (feat - mean) / std
    T = feat.shape[0]
    rec_n = np.zeros_like(feat_n); cnt = np.zeros(T)
    starts = list(range(0, max(1, T - window + 1), stride)) or [0]
    if starts[-1] + window < T: starts.append(T - window)
    for s in starts:
        w = feat_n[s:s+window]
        if len(w) < window: w = np.pad(w, ((0, window-len(w)), (0, 0)))
        with torch.no_grad():
            r = vae(torch.from_numpy(w).unsqueeze(0))["rec_pose"].squeeze(0).numpy()
        a = min(window, T - s); rec_n[s:s+a] += r[:a]; cnt[s:s+a] += 1
    return (rec_n / np.maximum(cnt, 1)[:, None]) * std + mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--dataset_dir", default="stage2/out/g1_dataset_gated8")
    p.add_argument("--clip_dir", default="stage2/out/g1_dataset_T4")  # full-clip features
    p.add_argument("--clips", nargs="+", default=None)
    p.add_argument("--out", default="")
    args = p.parse_args()
    vae = load_vae(args.vae_ckpt); mean, std = load_norm(args.dataset_dir)

    def find_clip(name):
        for sp in ("train", "test", "val"):
            f = os.path.join(args.clip_dir, sp, name + ".npz")
            if os.path.isfile(f): return f
        return None

    clips = args.clips or [f[:-4] for sp in ("train", "test", "val")
                           if os.path.isdir(os.path.join(args.clip_dir, sp))
                           for f in sorted(os.listdir(os.path.join(args.clip_dir, sp))) if f.endswith(".npz")]
    rows = {}
    print(f"{'clip':30s} {'jointRMSE':>9} {'leg':>6} {'arm':>6} {'waist':>6} {'<0.10':>6}")
    allj = []
    for c in clips:
        f = find_clip(c)
        if not f: continue
        feat = np.load(f, allow_pickle=True)["motion"].astype(np.float32)
        rec = reconstruct(vae, feat, mean, std)
        je = rec[:, 12:] - feat[:, 12:]                     # joint angle error [T,29]
        jrmse = float(np.sqrt((je**2).mean()))
        leg = float(np.sqrt((je[:, LEG]**2).mean())); arm = float(np.sqrt((je[:, ARM]**2).mean()))
        wst = float(np.sqrt((je[:, WAIST]**2).mean()))
        rows[c] = {"joint_rmse_rad": round(jrmse, 4), "leg_rmse": round(leg, 4),
                   "arm_rmse": round(arm, 4), "waist_rmse": round(wst, 4), "pass_0.10": jrmse < 0.10}
        allj.append(jrmse)
        print(f"{c:30s} {jrmse:9.4f} {leg:6.3f} {arm:6.3f} {wst:6.3f} {'Y' if jrmse<0.10 else 'n':>6}")
    mean_j = float(np.mean(allj)) if allj else float("nan")
    print(f"{'MEAN':30s} {mean_j:9.4f}   (paper biomech target < 0.10 rad)")
    if args.out:
        json.dump({"vae_ckpt": args.vae_ckpt, "mean_joint_rmse": round(mean_j, 4), "clips": rows},
                  open(args.out, "w"), indent=2)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
