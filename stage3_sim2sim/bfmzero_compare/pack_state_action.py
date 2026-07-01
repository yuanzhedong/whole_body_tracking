"""Pack the collected state-action trajectories (DATA_PIPELINE.md §6 top-level artifacts).

Reads <root>/{onpolicy,teacher_forced}/traj_*.npz and writes at <root>:
  * normalization.npz  — mean/std over qpos/qvel/obs_state/action/z/ref_dof_pos/ref_dof_vel
  * manifest.json      — per-trajectory {clip, mode, T, survival}
  * dataset_stats.json — total transitions, mean survival, per-mode counts

Pure numpy; run in any env.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

NORM_KEYS = ["qpos", "qvel", "obs_state", "action", "z", "ref_dof_pos", "ref_dof_vel"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--modes", nargs="+", default=["onpolicy", "teacher_forced"])
    args = p.parse_args()
    root = Path(args.root)

    manifest, sums, sqs, counts = [], {}, {}, {k: 0 for k in NORM_KEYS}
    per_mode = {}
    total_T = 0
    for mode in args.modes:
        files = sorted(glob.glob(str(root / mode / "traj_*.npz")))
        per_mode[mode] = {"trajectories": len(files), "transitions": 0, "survival_sum": 0.0}
        for f in files:
            d = np.load(f, allow_pickle=True)
            T = int(d["action"].shape[0])
            surv = float(d["alive"].mean()) if "alive" in d.files else float("nan")
            manifest.append({"file": Path(f).name, "clip": str(d["clip"]), "mode": mode,
                             "T": T, "survival": round(surv, 4)})
            per_mode[mode]["transitions"] += T
            per_mode[mode]["survival_sum"] += surv
            total_T += T
            for k in NORM_KEYS:
                if k not in d.files:
                    continue
                x = d[k].reshape(d[k].shape[0], -1).astype(np.float64)
                sums.setdefault(k, np.zeros(x.shape[1]))
                sqs.setdefault(k, np.zeros(x.shape[1]))
                sums[k] += x.sum(0); sqs[k] += (x ** 2).sum(0); counts[k] += x.shape[0]

    norm = {}
    for k in NORM_KEYS:
        if counts[k] == 0:
            continue
        mean = sums[k] / counts[k]
        var = np.maximum(sqs[k] / counts[k] - mean ** 2, 1e-12)
        norm[f"{k}_mean"] = mean.astype(np.float32); norm[f"{k}_std"] = np.sqrt(var).astype(np.float32)
    np.savez(root / "normalization.npz", **norm)
    json.dump(manifest, open(root / "manifest.json", "w"), indent=2)

    stats = {"total_trajectories": len(manifest), "total_transitions": total_T,
             "per_mode": {m: {"trajectories": v["trajectories"], "transitions": v["transitions"],
                              "mean_survival": round(v["survival_sum"] / max(1, v["trajectories"]), 4)}
                          for m, v in per_mode.items()},
             "mean_survival_all": round(
                 sum(v["survival_sum"] for v in per_mode.values()) /
                 max(1, sum(v["trajectories"] for v in per_mode.values())), 4)}
    json.dump(stats, open(root / "dataset_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"-> {root}/normalization.npz, manifest.json, dataset_stats.json")


if __name__ == "__main__":
    main()
