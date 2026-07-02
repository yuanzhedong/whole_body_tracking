"""Phase-2: train the BFM-Zero-distilled control VAE on collected (s, r, a*, z) pairs.

Consumes the phase-1 dataset (stage2/collect_bfmzero_pairs.py -> pairs_<mid>.npz) and
trains stage2/vae_model.MotionVAE:

    z_hat ~ E(ref_window[H frames])          # encoder, H = --horizon (sweepable here)
    a_hat = D(z_hat, proprio)                # decoder, state-conditioned
    loss  = || a_hat - a* ||^2  +  beta*KL(q(z)||N(0,I))  [+ align*|| adapt(z_hat) - z_bfm ||^2]

The encoder horizon H is a TRAINING-TIME knob (PLAN sec. 3): `ref` is logged per step,
so we slice an H-frame window on the fly -> ref_dim = H * ref_frame_dim. Sweep H without
re-collecting. Windows are clip-aware (never cross a motion boundary; tail is edge-padded).

Reports the G1/G3 gate metrics: recon RMSE, active latent dims, z-ablation. Runs in a
plain torch env (.venv6); no simulator needed. Real closed-loop (G2) is phase 3.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae_model import MotionVAE, DualHeadVAE, kl_divergence  # noqa: E402


def load_pairs(data_dir):
    """Read all pairs_<mid>.npz -> list of per-motion dicts (proprio, ref, action, z)."""
    files = sorted(glob.glob(os.path.join(data_dir, "pairs_*.npz")))
    if not files:
        raise FileNotFoundError(f"no pairs_*.npz in {data_dir} (run collect_bfmzero_pairs.py first)")
    motions = []
    for f in files:
        d = np.load(f)
        motions.append({k: np.asarray(d[k], np.float32) for k in ("proprio", "ref", "action", "z")})
    return motions, files


def build_windows(motions, horizon):
    """Per-step samples. ref_window = ref[t : t+H] (edge-padded at the clip tail), flattened.

    Future-facing window: the encoder sees the near-term reference intent to track. Returns
    X_ref[N, H*dr], X_pro[N, ds], Y_act[N, da], Z_bfm[N, dz], mids[N] (motion index for split).
    """
    Xr, Xp, Ya, Zb, mids = [], [], [], [], []
    for mi, m in enumerate(motions):
        ref, pro, act, z = m["ref"], m["proprio"], m["action"], m["z"]
        T, dr = ref.shape
        for t in range(T):
            idx = np.minimum(np.arange(t, t + horizon), T - 1)   # edge-pad the tail
            Xr.append(ref[idx].reshape(-1))
            Xp.append(pro[t]); Ya.append(act[t]); Zb.append(z[t]); mids.append(mi)
    return (np.stack(Xr).astype(np.float32), np.stack(Xp).astype(np.float32),
            np.stack(Ya).astype(np.float32), np.stack(Zb).astype(np.float32),
            np.asarray(mids, np.int64))


def standardize_fit(x):
    mu = x.mean(0); sd = np.clip(x.std(0), 1e-6, None)
    return mu.astype(np.float32), sd.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="dir of pairs_<mid>.npz from phase 1")
    p.add_argument("--out", default="stage2/out/control_vae")
    p.add_argument("--horizon", type=int, default=16, help="encoder reference window H (frames)")
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--beta", type=float, default=0.01, help="KL coef (Table S6 default)")
    p.add_argument("--align-coef", type=float, default=0.0,
                   help=">0 ties z_hat to the BFM-Zero latent via a linear adapter")
    p.add_argument("--motion-coef", type=float, default=0.0,
                   help=">0 adds a motion-reconstruction head M(z)->ref (DualHeadVAE); forces z to "
                        "encode motion (teacher-independent). 0 = single-head MotionVAE.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction of MOTIONS held out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    motions, files = load_pairs(args.data_dir)
    Xr, Xp, Ya, Zb, mids = build_windows(motions, args.horizon)
    dr_frame = motions[0]["ref"].shape[1]
    print(f"loaded {len(motions)} motions, {len(Xr)} samples | H={args.horizon} "
          f"ref_dim={Xr.shape[1]} ({dr_frame}/frame) proprio={Xp.shape[1]} act={Ya.shape[1]} z_bfm={Zb.shape[1]}")

    # split by motion (no window leakage)
    n_mot = len(motions)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_mot)
    n_val = max(1, int(round(args.val_frac * n_mot)))
    val_mot = set(perm[:n_val].tolist())
    is_val = np.array([m in val_mot for m in mids])
    tr, va = ~is_val, is_val
    print(f"split: {n_mot - n_val} train / {n_val} val motions | {tr.sum()} / {va.sum()} samples")

    # standardize inputs + target on TRAIN
    rmu, rsd = standardize_fit(Xr[tr]); pmu, psd = standardize_fit(Xp[tr]); amu, asd = standardize_fit(Ya[tr])

    def norm(x, mu, sd): return (x - mu) / sd
    to = lambda a: torch.tensor(a, device=dev)
    Xr_t, Xp_t, Ya_t = to(norm(Xr, rmu, rsd)), to(norm(Xp, pmu, psd)), to(norm(Ya, amu, asd))
    Zb_t = to(Zb)
    tr_t, va_t = to(tr), to(va)

    dual = args.motion_coef > 0
    if dual:
        model = DualHeadVAE(ref_dim=Xr.shape[1], proprio_dim=Xp.shape[1], act_dim=Ya.shape[1],
                            latent=args.latent).to(dev)
    else:
        model = MotionVAE(ref_dim=Xr.shape[1], proprio_dim=Xp.shape[1], act_dim=Ya.shape[1],
                          latent=args.latent).to(dev)
    adapter = nn.Linear(args.latent, Zb.shape[1]).to(dev) if args.align_coef > 0 else None
    params = list(model.parameters()) + (list(adapter.parameters()) if adapter else [])
    opt = torch.optim.AdamW(params, lr=args.lr)

    tr_idx = torch.nonzero(tr_t).squeeze(1)
    asd_t = to(asd)
    for ep in range(args.epochs):
        model.train()
        perm_i = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        tot = {"rec": 0.0, "kl": 0.0, "mo": 0.0, "n": 0}
        for s in range(0, len(perm_i), args.batch_size):
            b = perm_i[s:s + args.batch_size]
            if dual:
                ah, rhat, mu, logvar = model(Xr_t[b], Xp_t[b], sample=True)
                mo = ((rhat - Xr_t[b]) ** 2).mean()
            else:
                ah, mu, logvar = model(Xr_t[b], Xp_t[b], sample=True)
                mo = torch.tensor(0.0, device=dev)
            rec = ((ah - Ya_t[b]) ** 2).mean()
            kl = kl_divergence(mu, logvar)
            loss = rec + args.beta * kl + args.motion_coef * mo
            if adapter is not None:
                loss = loss + args.align_coef * ((adapter(mu) - Zb_t[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            bs = len(b)
            tot["rec"] += rec.item() * bs; tot["kl"] += kl.item() * bs
            tot["mo"] += float(mo.detach()) * bs; tot["n"] += bs
        if ep % 10 == 0 or ep == args.epochs - 1:
            m = evaluate(model, Xr_t, Xp_t, Ya_t, va_t, asd_t, args, dual)
            print(f"ep {ep:3d} | train rec {tot['rec']/tot['n']:.4f} kl {tot['kl']/tot['n']:.3f} "
                  f"mo {tot['mo']/tot['n']:.4f} | val rec_rmse {m['val_rec_rmse_raw']:.4f} "
                  f"active {m['active_dims']}/{args.latent} z_abl {m['z_ablation']:.3f} "
                  f"mo_rmse {m['motion_rmse']:.3f} mo_zabl {m['motion_z_ablation']:.3f}")

    metrics = evaluate(model, Xr_t, Xp_t, Ya_t, va_t, asd_t, args, dual)
    torch.save({"state_dict": model.state_dict(),
                "arch": {"ref_dim": Xr.shape[1], "proprio_dim": Xp.shape[1],
                         "act_dim": Ya.shape[1], "latent": args.latent, "horizon": args.horizon,
                         "dual_head": dual}},
               out / "control_vae.pt")
    np.savez(out / "normalization.npz", ref_mu=rmu, ref_sd=rsd, pro_mu=pmu, pro_sd=psd,
             act_mu=amu, act_sd=asd)
    json.dump({**metrics, "args": vars(args), "n_motions": n_mot, "n_samples": int(len(Xr))},
              open(out / "metrics.json", "w"), indent=2)
    print("G1 recon_rmse(raw)=%.4f | G3 active=%d/%d z_ablation=%.3f | motion_rmse=%.3f "
          "motion_z_ablation=%.3f -> %s" % (
              metrics["val_rec_rmse_raw"], metrics["active_dims"], args.latent, metrics["z_ablation"],
              metrics["motion_rmse"], metrics["motion_z_ablation"], out / "control_vae.pt"))


@torch.no_grad()
def evaluate(model, Xr, Xp, Ya, va_mask, asd_t, args, dual=False):
    """G1 action recon RMSE + G3 active dims + z-ablation. Dual-head adds motion metrics."""
    model.eval()
    idx = torch.nonzero(va_mask).squeeze(1)
    if len(idx) == 0:
        idx = torch.arange(min(2048, Xr.shape[0]), device=Xr.device)
    if dual:
        ah, rhat, mu, logvar = model(Xr[idx], Xp[idx], sample=False)
    else:
        ah, mu, logvar = model(Xr[idx], Xp[idx], sample=False)
    rec_norm = torch.sqrt(((ah - Ya[idx]) ** 2).mean()).item()
    rec_raw = torch.sqrt(((ah - Ya[idx]) ** 2 * asd_t ** 2).mean()).item()  # de-standardized
    kl_dim = (0.5 * (mu ** 2 + logvar.exp() - logvar - 1)).mean(0)
    active = int((kl_dim > 0.01).sum().item())
    z0 = torch.zeros_like(mu)
    a_z0 = model.decode_action(z0, Xp[idx]) if dual else model.decode(z0, Xp[idx])
    z_abl = (torch.norm(ah - a_z0, dim=-1).mean() / (torch.norm(ah, dim=-1).mean() + 1e-6)).item()
    # dual-head: does z encode the MOTION? recon quality + how much motion depends on z
    motion_rmse, motion_z_abl = 0.0, 0.0
    if dual:
        motion_rmse = torch.sqrt(((rhat - Xr[idx]) ** 2).mean()).item()
        r_z0 = model.decode_motion(z0)
        motion_z_abl = (torch.norm(rhat - r_z0, dim=-1).mean() / (torch.norm(rhat, dim=-1).mean() + 1e-6)).item()
    return {"val_rec_rmse_norm": rec_norm, "val_rec_rmse_raw": rec_raw, "active_dims": active,
            "z_ablation": z_abl, "motion_rmse": motion_rmse, "motion_z_ablation": motion_z_abl}


if __name__ == "__main__":
    main()
