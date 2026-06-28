"""Log VAE ablation results to W&B as a unified table + summary run.

Creates:
  - One W&B run per tier (g1-vae-ablation project) with Phase-0 + Phase-2 metrics
  - A summary run with a comparison table and scaling curve
"""
import argparse, json, os, glob
import wandb

TIER_CLIPS = {
    "T1": ["walk1_subject1"],
    "T2": ["walk1_subject1", "lafan_run1_subject2", "lafan_sprint1_subject2"],
    "T3": ["walk1_subject1", "lafan_run1_subject2", "lafan_sprint1_subject2",
           "lafan_dance1_subject1", "lafan_dance2_subject1"],
    "T4": ["walk1_subject1", "lafan_run1_subject2", "lafan_sprint1_subject2",
           "lafan_dance1_subject1", "lafan_dance2_subject1",
           "lafan_jumps1_subject1", "lafan_fallAndGetUp1_subject1",
           "lafan_fight1_subject2", "lafan_fightAndSports1_subject1"],
}
TIER_LABELS = {
    "T1": "walk only",
    "T2": "locomotion (walk+run+sprint)",
    "T3": "locomotion + dance",
    "T4": "all categories",
}


def load_manifest(dataset_dir):
    p = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))


def count_windows(dataset_dir, window_size=128, step_size=32):
    total = 0
    for split in ["train", "val", "test"]:
        for f in glob.glob(os.path.join(dataset_dir, split, "*.npz")):
            import numpy as np
            d = np.load(f)
            n = d["motion"].shape[0]
            if n >= window_size:
                total += max(1, (n - window_size) // step_size + 1)
            else:
                total += 1
    return total


def load_phase0(out_dir, tier):
    p = os.path.join(out_dir, f"phase0_{tier}.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))


def load_sim2sim(out_dir, tier):
    p = os.path.join(out_dir, f"sim2sim_{tier}.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiers", nargs="+", default=["T1", "T2", "T3", "T4"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--project", default="g1-vae-ablation")
    p.add_argument("--entity", default="cs224n-robustqa")
    args = p.parse_args()

    table_rows = []

    for tier in args.tiers:
        dataset_dir = os.path.join(args.out_dir, f"g1_dataset_{tier}")
        phase0 = load_phase0(args.out_dir, tier)
        sim2sim = load_sim2sim(args.out_dir, tier)
        n_clips = len(TIER_CLIPS.get(tier, []))

        try:
            n_windows = count_windows(dataset_dir)
        except Exception:
            n_windows = -1

        rmse = phase0.get("phase0_rmse_rad", phase0.get("joint_angle_rmse", None))
        root_err = phase0.get("root_orient_error_deg", None)
        val_loss = phase0.get("val_loss", None)
        survival_vae = None
        survival_teacher = None
        if sim2sim:
            vae_r = sim2sim.get("G2_closed_loop_vae", {})
            teacher_r = sim2sim.get("G2_closed_loop_teacher", {})
            survival_vae = vae_r.get("survival", vae_r.get("survival_rate"))
            survival_teacher = teacher_r.get("survival", teacher_r.get("survival_rate"))

        # Per-tier W&B run
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=f"ablation_{tier}",
            tags=["ablation", tier, "g1_vae"],
            config={
                "tier": tier,
                "label": TIER_LABELS.get(tier, tier),
                "n_clips": n_clips,
                "clips": TIER_CLIPS.get(tier, []),
                "n_windows": n_windows,
            },
        )
        metrics = {"n_clips": n_clips, "n_windows": n_windows}
        if rmse is not None:
            metrics["phase0/joint_angle_rmse_rad"] = float(rmse)
        if root_err is not None:
            metrics["phase0/root_orient_error_deg"] = float(root_err)
        if val_loss is not None:
            metrics["phase0/val_loss"] = float(val_loss)
        if survival_vae is not None:
            metrics["phase2/survival_vae"] = float(survival_vae)
        if survival_teacher is not None:
            metrics["phase2/survival_teacher"] = float(survival_teacher)
            metrics["phase2/survival_ratio"] = float(survival_vae) / max(float(survival_teacher), 1e-6)
        wandb.log(metrics)
        wandb.finish()

        table_rows.append({
            "tier": tier,
            "label": TIER_LABELS.get(tier, tier),
            "n_clips": n_clips,
            "n_windows": n_windows,
            "joint_RMSE_rad": round(float(rmse), 4) if rmse is not None else "—",
            "root_orient_err_deg": round(float(root_err), 1) if root_err is not None else "—",
            "val_loss": round(float(val_loss), 4) if val_loss is not None else "—",
            "sim2sim_survival_vae": round(float(survival_vae), 3) if survival_vae is not None else "—",
            "sim2sim_survival_teacher": round(float(survival_teacher), 3) if survival_teacher is not None else "—",
        })

    # Summary run with comparison table
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name="ablation_summary",
        tags=["summary", "ablation", "g1_vae"],
    )
    cols = ["tier", "label", "n_clips", "n_windows",
            "joint_RMSE_rad", "root_orient_err_deg", "val_loss",
            "sim2sim_survival_vae", "sim2sim_survival_teacher"]
    tbl = wandb.Table(columns=cols)
    for row in table_rows:
        tbl.add_data(*[row[c] for c in cols])
    wandb.log({"ablation_results": tbl})

    # Scaling curve: RMSE vs n_windows
    for row in table_rows:
        if isinstance(row["joint_RMSE_rad"], float) and row["n_windows"] > 0:
            wandb.log({
                "scaling/joint_RMSE_rad": row["joint_RMSE_rad"],
                "scaling/n_windows": row["n_windows"],
                "scaling/n_clips": row["n_clips"],
            })
    wandb.finish()

    print("\n=== ABLATION SUMMARY ===")
    for row in table_rows:
        print(f"  {row['tier']} ({row['label']}): "
              f"clips={row['n_clips']} windows={row['n_windows']} "
              f"RMSE={row['joint_RMSE_rad']} survival={row['sim2sim_survival_vae']}")
    print(f"\nW&B project: https://wandb.ai/{args.entity}/{args.project}")


if __name__ == "__main__":
    main()
