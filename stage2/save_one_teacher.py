"""Save ONE teacher policy to W&B as a reloadable artifact. Used by amass_teacher_queue.sh after each
train, so policies are preserved before the local run dir is deleted (disk hygiene over thousands of clips).
Usage: save_one_teacher.py <clip> <run_dir> <iter>   (artifact teacher_<clip>, project g1-teachers)
"""
import os, sys, wandb

ENTITY = "cs224n-robustqa"; PROJECT = "g1-teachers"

def main():
    clip, rd, it = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ck = f"{rd}/model_{it}.pt"
    if not os.path.exists(ck):
        print("no ckpt", ck); sys.exit(1)
    run = wandb.init(entity=ENTITY, project=PROJECT, name=f"teacher_{clip}", reinit=True,
                     config={"clip": clip, "iters": it + 1, "source": "amass"})
    art = wandb.Artifact(f"teacher_{clip}", type="model",
                         metadata={"clip": clip, "iters": it + 1, "source": "amass"})
    art.add_file(ck, name=f"model_{it}.pt")
    for f in ("agent.yaml", "agent.pkl", "env.yaml", "env.pkl"):
        p = f"{rd}/params/{f}"
        if os.path.exists(p): art.add_file(p, name=f"params/{f}")
    run.log_artifact(art); run.finish()
    print("saved", clip)

if __name__ == "__main__":
    main()
