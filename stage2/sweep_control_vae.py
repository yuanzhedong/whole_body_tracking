"""Phase-2 hyperparameter sweep for the BFM-Zero control-VAE, fanned across GPUs.

Runs a grid of (horizon H, latent, beta, align) training jobs (stage2/train_control_vae.py)
concurrently over a pool of GPU slots, then collates every run's metrics.json into a
leaderboard sorted by G3 z-ablation (latent-usage) then G1 recon RMSE.

Each training job is small (an MLP on the (s,r,a*,z) pairs), so many fit per GPU. Runs
the trainer via .venv6 (torch 2.10/cu128; works on Blackwell + 4090).

Example:
    .venv6/bin/python stage2/sweep_control_vae.py \\
        --data-dir stage2/out/bfmpairs_seed40_dart01 \\
        --out stage2/out/sweep_seed40 --gpus 1,5 --slots-per-gpu 2 --epochs 80
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

PY = os.environ.get("SWEEP_PY", ".venv6/bin/python")

# Default grid — deliberately broad (compute is cheap here). Overridable via CLI.
# motion_coef=0 -> single-head (action-only) MotionVAE; >0 -> dual-head (+ motion-recon
# head). Including 0 in the list tests BOTH approaches in one sweep.
HORIZON = [1, 8, 16, 32]
LATENT = [16, 32, 64]
BETA = [1e-3, 1e-2, 1e-1]
ALIGN = [0.0, 0.5]
MOTION = [0.0]


def parse_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def grid(H_, L_, B_, A_, M_):
    for H, L, b, a, m in itertools.product(H_, L_, B_, A_, M_):
        yield {"horizon": H, "latent": L, "beta": b, "align_coef": a, "motion_coef": m}


def tag(c):
    return f"H{c['horizon']}_L{c['latent']}_b{c['beta']:g}_a{c['align_coef']:g}_m{c['motion_coef']:g}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpus", default="1,5", help="comma list of GPU indices (PCI order)")
    p.add_argument("--slots-per-gpu", type=int, default=2)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--horizons", default=None, help="comma list; default 1,8,16,32")
    p.add_argument("--latents", default=None, help="comma list; default 16,32,64")
    p.add_argument("--betas", default=None, help="comma list; default 1e-3,1e-2,1e-1")
    p.add_argument("--aligns", default=None, help="comma list; default 0,0.5")
    p.add_argument("--motion-coefs", default=None,
                   help="comma list; default 0 (single-head). Add >0 for dual-head, e.g. 0,0.1,1,10")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    slots = gpus * args.slots_per_gpu           # one entry per concurrent worker
    H_ = parse_floats(args.horizons) if args.horizons else HORIZON
    H_ = [int(x) for x in H_]
    L_ = [int(x) for x in parse_floats(args.latents)] if args.latents else LATENT
    B_ = parse_floats(args.betas) if args.betas else BETA
    A_ = parse_floats(args.aligns) if args.aligns else ALIGN
    M_ = parse_floats(args.motion_coefs) if args.motion_coefs else MOTION
    configs = list(grid(H_, L_, B_, A_, M_))
    print(f"sweep: {len(configs)} configs over {len(slots)} slots (GPUs {gpus} x{args.slots_per_gpu}) "
          f"| data={args.data_dir}")

    running = {}   # slot_gpu -> (Popen, tag, logfile-handle)
    queue = list(configs)
    slot_free = list(slots)
    done = 0

    def launch(gpu, c):
        run_out = out / tag(c)
        run_out.mkdir(parents=True, exist_ok=True)
        logf = open(run_out / "train.log", "w")
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=".")
        cmd = [PY, "stage2/train_control_vae.py",
               "--data-dir", args.data_dir, "--out", str(run_out),
               "--horizon", str(c["horizon"]), "--latent", str(c["latent"]),
               "--beta", str(c["beta"]), "--align-coef", str(c["align_coef"]),
               "--motion-coef", str(c["motion_coef"]),
               "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--lr", str(args.lr), "--val-frac", str(args.val_frac), "--device", "cuda"]
        return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env), logf

    while queue or running:
        # fill free slots
        while slot_free and queue:
            gpu = slot_free.pop(0)
            c = queue.pop(0)
            proc, logf = launch(gpu, c)
            running[id(proc)] = (proc, gpu, tag(c), logf)
            print(f"  [{len(configs)-len(queue)}/{len(configs)}] start {tag(c)} on GPU{gpu}")
        # reap finished
        for key, (proc, gpu, t, logf) in list(running.items()):
            if proc.poll() is not None:
                logf.close(); slot_free.append(gpu); running.pop(key); done += 1
                ok = (out / t / "metrics.json").exists()
                print(f"  done {t} rc={proc.returncode} metrics={'OK' if ok else 'MISSING'} ({done}/{len(configs)})")
        time.sleep(2)

    collate(out)


def collate(out: Path):
    rows = []
    for md in sorted(out.glob("*/metrics.json")):
        m = json.load(open(md))
        a = m.get("args", {})
        rows.append({"tag": md.parent.name, "horizon": a.get("horizon"), "latent": a.get("latent"),
                     "beta": a.get("beta"), "align": a.get("align_coef"), "motion": a.get("motion_coef"),
                     "head": "dual" if (a.get("motion_coef") or 0) > 0 else "single",
                     "recon_rmse_raw": round(m.get("val_rec_rmse_raw", float("nan")), 5),
                     "active_dims": m.get("active_dims"), "z_ablation": round(m.get("z_ablation", 0), 4),
                     "motion_rmse": round(m.get("motion_rmse", 0), 4),
                     "motion_z_ablation": round(m.get("motion_z_ablation", 0), 4),
                     "n_motions": m.get("n_motions"), "n_samples": m.get("n_samples")})
    if not rows:
        print("no metrics.json found — all runs failed? check */train.log")
        return
    # rank: latent must be USED (max of action- and motion-z-ablation) AND action recon good
    def zu(r): return max(r["z_ablation"] or 0, r["motion_z_ablation"] or 0)
    rows.sort(key=lambda r: (-zu(r), r["recon_rmse_raw"]))
    csv_path = out / "leaderboard.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n=== leaderboard ({len(rows)} runs) -> {csv_path} ===")
    print(f"{'tag':34s} {'head':>6s} {'a_rmse':>7s} {'z_abl':>6s} {'mo_rmse':>7s} {'mo_zabl':>7s}")
    for r in rows[:15]:
        print(f"{r['tag']:34s} {r['head']:>6s} {r['recon_rmse_raw']:7.4f} {r['z_ablation']:6.3f} "
              f"{r['motion_rmse']:7.3f} {r['motion_z_ablation']:6.3f}")


if __name__ == "__main__":
    main()
