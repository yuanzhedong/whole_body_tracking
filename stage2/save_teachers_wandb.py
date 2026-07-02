"""Save per-clip tracking-policy (teacher) checkpoints to W&B as reloadable artifacts.
Each artifact `teacher_<clip>` (type=model, project cs224n-robustqa/g1-teachers) packs model_11999.pt +
params/{agent,env}.{yaml,pkl}. Metadata carries gate survival (if known) + iters. Idempotent-ish: logs a
new version each run. CPU only (file upload). Usage: .venv/bin/python stage2/save_teachers_wandb.py [clip ...]
"""
import os, glob, sys, json
import wandb

ENTITY = "cs224n-robustqa"; PROJECT = "g1-teachers"; BASE = "logs/rsl_rl/g1_flat"
CKPT = "model_11999.pt"

def gate_map():
    m = {}
    p = "stage2/out/gate_results.txt"
    if os.path.exists(p):
        for l in open(p):
            t = l.split()
            if len(t) >= 2 and t[1].startswith("survival="):
                m[t[0]] = t[1].split("=")[1]
    return m

def latest_run(clip):
    dirs = sorted([d for d in glob.glob(f"{BASE}/*teacher_{clip}") if d.endswith(f"teacher_{clip}")],
                  key=os.path.getmtime, reverse=True)
    for d in dirs:
        if os.path.exists(f"{d}/{CKPT}"):
            return d
    return None

def main():
    gates = gate_map()
    if len(sys.argv) > 1:
        clips = sys.argv[1:]
    else:
        clips = sorted(set(d.split("teacher_")[1] for d in glob.glob(f"{BASE}/*teacher_*")
                           if "teacher_" in os.path.basename(d)))
    print(f"{len(clips)} clips to save")
    done = 0
    for clip in clips:
        rd = latest_run(clip)
        if not rd:
            print("SKIP", clip, "(no model_11999.pt)"); continue
        surv = gates.get(clip, "NA")
        run = wandb.init(entity=ENTITY, project=PROJECT, name=f"teacher_{clip}", reinit=True,
                         config={"clip": clip, "iters": 12000, "gate_survival": surv})
        art = wandb.Artifact(f"teacher_{clip}", type="model",
                             metadata={"clip": clip, "iters": 12000, "gate_survival": surv,
                                       "gate_pass": (surv != "NA" and float(surv) >= 0.95)})
        art.add_file(f"{rd}/{CKPT}", name="model_11999.pt")
        for f in ("agent.yaml", "agent.pkl", "env.yaml", "env.pkl"):
            p = f"{rd}/params/{f}"
            if os.path.exists(p): art.add_file(p, name=f"params/{f}")
        run.log_artifact(art); run.finish()
        done += 1; print(f"saved teacher_{clip}  gate={surv}")
    print(f"DONE: {done}/{len(clips)} saved to {ENTITY}/{PROJECT}")

if __name__ == "__main__":
    main()
