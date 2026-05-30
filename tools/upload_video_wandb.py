"""Upload a rendered MP4 to an existing WandB run so it shows on the dashboard.
Usage: python upload_video_wandb.py --run <entity/project/run_id> --video out.mp4 [--key media/policy] [--caption "..."]"""
import argparse, wandb

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="entity/project/run_id")
p.add_argument("--video", required=True)
p.add_argument("--key", default="media/policy_render")
p.add_argument("--caption", default="")
p.add_argument("--fps", type=int, default=25)
a = p.parse_args()

entity, project, run_id = a.run.split("/")
run = wandb.init(entity=entity, project=project, id=run_id, resume="must")
run.log({a.key: wandb.Video(a.video, caption=a.caption, fps=a.fps, format="mp4")})
run.finish()
print(f"UPLOADED {a.video} -> {a.run} [{a.key}]")
