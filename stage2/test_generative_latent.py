"""Is the dual-head latent generative / diffusion-ready? (offline, no simulator)

Loads a trained VAE checkpoint and measures three things the diffusion stage needs:
  1. Aggregate posterior ~ N(0,I): per-dim mean/std of the encoded mu over real windows.
     (diffusion samples z~N(0,I); if q(z) is far from N(0,I) the samples are OOD.)
  2. Prior-sample plausibility: sample z~N(0,I), decode the motion head M(z), and compare
     the feature-wise std of prior-decoded motion vs real motion. Close => in-distribution.
  3. Interpolation smoothness: encode two clips, interpolate z, decode M(z); mean step L2.
     Smooth interpolation => a well-formed motion manifold.

Single-head checkpoints have no motion head, so 2-3 are skipped (only posterior stats).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae_model import MotionVAE, DualHeadVAE  # noqa: E402


def load(ckpt_dir):
    ck = torch.load(Path(ckpt_dir) / "control_vae.pt", map_location="cpu", weights_only=False)
    a = ck["arch"]; dual = bool(a.get("dual_head", False))
    Cls = DualHeadVAE if dual else MotionVAE
    m = Cls(ref_dim=a["ref_dim"], proprio_dim=a["proprio_dim"], act_dim=a["act_dim"], latent=a["latent"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    nz = np.load(Path(ckpt_dir) / "normalization.npz")
    return m, a["horizon"], {k: nz[k].astype(np.float32) for k in nz.files}, dual


def ref_windows(data_dir, H, n_max=4000):
    """Real normalized-ready ref windows [N, H*dr] from BC pairs."""
    W, clips = [], []
    for f in sorted(glob.glob(str(Path(data_dir) / "pairs_*.npz"))):
        r = np.asarray(np.load(f)["ref"], np.float32); T = len(r)
        wins = [r[np.minimum(np.arange(t, t + H), T - 1)].reshape(-1) for t in range(T)]
        clips.append(np.stack(wins))
    allw = np.concatenate(clips, 0)
    if len(allw) > n_max:
        allw = allw[np.random.default_rng(0).choice(len(allw), n_max, replace=False)]
    return allw.astype(np.float32), clips


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    vae, H, norm, dual = load(args.ckpt_dir)
    W, clips = ref_windows(args.data_dir, H)
    Wn = (W - norm["ref_mu"]) / norm["ref_sd"]
    x = torch.tensor(Wn)
    mu, logvar = vae.encode(x)
    mu = mu.numpy()

    # 1. aggregate posterior ~ N(0,I)?
    dim_mean = mu.mean(0); dim_std = mu.std(0)
    active = int((dim_std > 0.1).sum())
    post = {"active_dims": active, "latent": vae.latent,
            "agg_mean_abs": round(float(np.abs(dim_mean).mean()), 4),      # ~0 ideal
            "agg_std_mean": round(float(dim_std.mean()), 4),               # ~1 ideal
            "agg_std_active_mean": round(float(dim_std[dim_std > 0.1].mean()) if active else 0.0, 4)}

    res = {"ckpt": args.ckpt_dir, "head": "dual" if dual else "single", "horizon": H, "posterior": post}

    if dual:
        # 2. prior-sample plausibility: z~N(0,I) -> M(z); std vs real
        z = torch.randn(2000, vae.latent)
        rhat = vae.decode_motion(z).numpy()
        real_std = Wn.std(0); prior_std = rhat.std(0)
        # how close is prior-decoded feature spread to real (1.0 = identical)
        ratio = prior_std / (real_std + 1e-6)
        res["prior_sample"] = {"std_ratio_mean": round(float(ratio.mean()), 3),
                               "std_ratio_median": round(float(np.median(ratio)), 3),
                               "frac_within_2x": round(float(((ratio > 0.5) & (ratio < 2)).mean()), 3)}
        # 3. interpolation smoothness (between the first windows of two clips)
        if len(clips) >= 2:
            a0 = (clips[0][0] - norm["ref_mu"]) / norm["ref_sd"]
            b0 = (clips[1][0] - norm["ref_mu"]) / norm["ref_sd"]
            za, _ = vae.encode(torch.tensor(a0)[None]); zb, _ = vae.encode(torch.tensor(b0)[None])
            ts = torch.linspace(0, 1, 11)[:, None]
            zpath = (1 - ts) * za + ts * zb
            mpath = vae.decode_motion(zpath).numpy()
            steps = np.linalg.norm(np.diff(mpath, axis=0), axis=1)
            res["interpolation"] = {"mean_step_L2": round(float(steps.mean()), 3),
                                    "max_step_L2": round(float(steps.max()), 3),
                                    "smoothness_ratio": round(float(steps.max() / (steps.mean() + 1e-6)), 2)}

    print(json.dumps(res, indent=2))
    out = args.out or (Path(args.ckpt_dir) / "generative_test.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
