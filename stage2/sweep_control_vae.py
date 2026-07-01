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

# Grid — deliberately broad (compute is cheap here).
HORIZON = [1, 8, 16, 32]
LATENT = [16, 32, 64]
BETA = [1e-3, 1e-2, 1e-1]
ALIGN = [0.0, 0.5]


def grid():
    for H, L, b, a in itertools.product(HORIZON, LATENT, BETA, ALIGN):
        yield {"horizon": H, "latent": L, "beta": b, "align_coef": a}


def tag(c):
    return f"H{c['horizon']}_L{c['latent']}_b{c['beta']:g}_a{c['align_coef']:g}"


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
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    slots = gpus * args.slots_per_gpu           # one entry per concurrent worker
    configs = list(grid())
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
                     "beta": a.get("beta"), "align": a.get("align_coef"),
                     "recon_rmse_raw": round(m.get("val_rec_rmse_raw", float("nan")), 5),
                     "active_dims": m.get("active_dims"), "z_ablation": round(m.get("z_ablation", 0), 4),
                     "n_motions": m.get("n_motions"), "n_samples": m.get("n_samples")})
    if not rows:
        print("no metrics.json found — all runs failed? check */train.log")
        return
    # rank: latent must be USED (z_ablation high) AND reconstruction good (rmse low)
    rows.sort(key=lambda r: (-(r["z_ablation"] or 0), r["recon_rmse_raw"]))
    csv_path = out / "leaderboard.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n=== leaderboard ({len(rows)} runs) -> {csv_path} ===")
    print(f"{'tag':28s} {'rmse':>8s} {'active':>6s} {'z_abl':>6s}")
    for r in rows[:12]:
        print(f"{r['tag']:28s} {r['recon_rmse_raw']:8.4f} {str(r['active_dims']):>6s} {r['z_ablation']:6.3f}")


if __name__ == "__main__":
    main()
