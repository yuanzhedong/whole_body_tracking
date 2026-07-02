"""Stateless overnight monitor. Run once per wakeup. Re-derives all state from disk.

For each registered experiment:
  - find latest checkpoint epoch
  - compute windowed val/train RMSE (via venv_umt subprocess)
  - record trajectory point
  - if run finished (epoch >= end-1 AND no live process), mark done + log final to W&B
Writes /tmp/overnight_progress.md and prints a summary table.

Run:  .venv/bin/python stage2/overnight_monitor.py
"""
import os, sys, glob, json, subprocess, datetime

WBT = "/ws/user/yzdong/src/github/whole_body_tracking"
EXP_ROOT = f"{WBT}/UniMoTok/experiments/biomechanics_tokenizer"
UMT_PY = f"{WBT}/UniMoTok/.venv_umt/bin/python"
STATE = "/tmp/overnight_state.json"
PROGRESS = "/tmp/overnight_progress.md"

# name -> (exp_subdir, data_dir, num_layers, ff_size, latent, end_epoch, cfgtag)
EXPERIMENTS = {
    # within_clip experiments (proper in-distribution val) — the real science
    "EX_T1w_base":  ("EX_T1w_base",  "g1_dataset_T1within", 5, 1024, 128, 20000),
    "EX_T2w_base":  ("EX_T2w_base",  "g1_dataset_T2within", 5, 1024, 128, 20000),
    "EX_T3w_base":  ("EX_T3w_base",  "g1_dataset_T3within", 5, 1024, 128, 20000),
    "EX_T4w_base":  ("EX_T4w_base",  "g1_dataset_T4within", 5, 1024, 128, 20000),
    "EX_T4w_big":   ("EX_T4w_big",   "g1_dataset_T4within", 9, 1536, 256, 20000),
    "EX_T4w_lowkl": ("EX_T4w_lowkl", "g1_dataset_T4within", 5, 1024, 128, 20000),
    # original 20k runs (flawed val, but track training convergence)
    "T1_old": ("G1_MldVAE_T1", "g1_dataset_T1", 5, 1024, 128, 20000),
    "T2_old": ("G1_MldVAE_T2", "g1_dataset_T2", 5, 1024, 128, 20000),
    "T3_old": ("G1_MldVAE_T3", "g1_dataset_T3", 5, 1024, 128, 20000),
    "T4_old": ("G1_MldVAE_T4", "g1_dataset_T4", 5, 1024, 128, 20000),
}


def latest_ckpt(exp_subdir):
    cs = [f for f in glob.glob(f"{EXP_ROOT}/{exp_subdir}/checkpoints/epoch=*.ckpt") if "-v" not in f]
    if not cs:
        return None, None
    best = max(cs, key=lambda p: int(os.path.basename(p).replace("epoch=", "").replace(".ckpt", "")))
    return best, int(os.path.basename(best).replace("epoch=", "").replace(".ckpt", ""))


def is_running(name):
    # crude: any train_tokenizer proc whose cmdline mentions this exp's config name
    try:
        out = subprocess.run(["pgrep", "-af", "train_tokenizer"], capture_output=True, text=True).stdout
        return name.replace("_old", "") in out or name in out
    except Exception:
        return False


def eval_rmse(ckpt, data_dir, nl, ff, lat):
    try:
        r = subprocess.run(
            [UMT_PY, f"{WBT}/stage2/eval_vae_rmse.py", "--ckpt", ckpt,
             "--data_dir", f"{WBT}/stage2/out/{data_dir}",
             "--num_layers", str(nl), "--ff_size", str(ff), "--latent", str(lat)],
            capture_output=True, text=True, timeout=300)
        for line in r.stdout.splitlines():
            if line.startswith("RMSE_JSON "):
                return json.loads(line[len("RMSE_JSON "):])
    except Exception as e:
        return {"error": str(e)[:80]}
    return None


def main():
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    now = datetime.datetime.now().strftime("%H:%M")
    rows = []
    newly_done = []

    for name, (sub, data, nl, ff, lat, end) in EXPERIMENTS.items():
        ckpt, epoch = latest_ckpt(sub)
        running = is_running(name)
        st = state.setdefault(name, {"traj": [], "done": False})
        if epoch is None:
            rows.append((name, "no ckpt", "-", "-", "-", running))
            continue
        rm = eval_rmse(ckpt, data, nl, ff, lat) or {}
        vr, tr = rm.get("val_rmse"), rm.get("train_rmse")
        st["traj"].append({"t": now, "epoch": epoch, "val": vr, "train": tr})
        st["traj"] = st["traj"][-50:]  # cap
        st["last_epoch"], st["last_val"], st["last_train"] = epoch, vr, tr

        done = (epoch >= end - 1) or (not running and epoch > 100)
        if done and not st["done"]:
            st["done"] = True
            newly_done.append((name, epoch, vr, tr, data, nl, ff, lat))
        rows.append((name, f"{epoch}/{end}", vr, tr,
                     round(vr-tr, 4) if vr and tr else "-", running))

    json.dump(state, open(STATE, "w"), indent=2)

    # progress md
    with open(PROGRESS, "w") as f:
        f.write(f"# Overnight VAE monitor — {now}\n\n")
        f.write("| exp | epoch | val_rmse | train_rmse | gap | live |\n|---|---|---|---|---|---|\n")
        for nm, ep, vr, tr, gap, run in rows:
            f.write(f"| {nm} | {ep} | {vr} | {tr} | {gap} | {'yes' if run else 'no'} |\n")
        if newly_done:
            f.write("\n**Newly completed this cycle:** " +
                    ", ".join(f"{n}(val={v})" for n, _, v, *_ in newly_done) + "\n")

    # print summary
    print(f"=== Overnight monitor {now} ===")
    print(f"{'exp':14s} {'epoch':12s} {'val':8s} {'train':8s} {'gap':8s} live")
    for nm, ep, vr, tr, gap, run in rows:
        print(f"{nm:14s} {str(ep):12s} {str(vr):8s} {str(tr):8s} {str(gap):8s} {'Y' if run else 'n'}")

    # log newly-done to W&B
    if newly_done:
        try:
            import wandb
            for name, epoch, vr, tr, data, nl, ff, lat in newly_done:
                run = wandb.init(entity="cs224n-robustqa", project="g1-vae-ablation",
                                 name=f"DONE_{name}", tags=["final", "overnight", name], reinit=True,
                                 config={"exp": name, "data": data, "num_layers": nl,
                                         "ff_size": ff, "latent": lat, "epoch": epoch})
                wandb.log({"final/val_rmse": vr, "final/train_rmse": tr,
                           "final/gap": (vr-tr) if vr and tr else None, "epoch": epoch})
                wandb.finish()
                print(f"  logged DONE_{name} to W&B (val={vr})")
        except Exception as e:
            print(f"  W&B log failed: {e}")

    print(f"\nProgress: {PROGRESS}")
    return newly_done


if __name__ == "__main__":
    main()
