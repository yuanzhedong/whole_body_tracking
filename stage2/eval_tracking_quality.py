"""Per-clip Stage-0 tracking quality eval for the G1 motion corpus.

Runs the Stage-0 tracking policy on each clip in artifacts/<name>:v0/motion.npz and
records per-clip survival rate + E_mpbpe_mm.  The output JSON feeds directly into
export_g1_motion.py --quality_json to filter low-quality clips before VAE training.

Usage (4090, headless):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \\
    .venv/bin/python stage2/eval_tracking_quality.py \\
      --teacher_ckpt logs/rsl_rl/g1_flat/<run>/model_29999.pt \\
      --artifacts_dir artifacts \\
      --out stage2/out/track_quality.json

Then re-export the dataset with the quality filter:
  .venv/bin/python stage2/export_g1_motion.py --artifacts_dir artifacts \\
      --out_dir stage2/out/g1_dataset --target_fps 20 --to_yup \\
      --quality_json stage2/out/track_quality.json --min_survival 0.95 --max_mpbpe_mm 50

Clip selection: skips amass_* (TODO: short/undertested) and any clip whose npz is
missing.  Runs each clip for --eval_reps full repetitions and averages the metrics.
"""
import argparse, os, sys, json, glob
sys.path.append("/ws/user/yzdong/src/github/whole_body_tracking/scripts/rsl_rl")
from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--teacher_ckpt", required=True)
parser.add_argument("--artifacts_dir", default="artifacts")
parser.add_argument("--eval_reps", type=int, default=2,
                    help="number of full-clip repetitions to average over per clip")
parser.add_argument("--out", default="stage2/out/track_quality.json")
parser.add_argument("--skip_amass", action="store_true", default=True,
                    help="skip amass_* clips (TODO); disable with --no-skip_amass")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import whole_body_tracking.tasks  # noqa: F401,E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

MET = ["error_body_pos", "error_joint_pos"]


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    env_cfg.scene.num_envs = args.num_envs

    paths = sorted(glob.glob(os.path.join(args.artifacts_dir, "*", "motion.npz")))
    if not paths:
        raise SystemExit(f"no motion.npz under {args.artifacts_dir}/*/")

    results = {}
    for path in paths:
        name = os.path.basename(os.path.dirname(path)).replace(":v0", "")
        if args.skip_amass and name.startswith("amass_"):
            print(f"  skip  {name} (amass TODO)"); continue

        env_cfg.commands.motion.motion_file = path
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.teacher_ckpt)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        cmd = env.unwrapped.command_manager.get_term("motion")
        tm = env.unwrapped.termination_manager
        mlen = int(cmd.motion.time_step_total)

        n_fail = n_comp = 0
        sums = {k: 0.0 for k in MET}
        csum = 0
        obs, _ = env.get_observations()
        for _ in range(mlen * args.eval_reps):
            with torch.inference_mode():
                a = policy(obs)
                obs, _, _, _ = env.step(a)
            n_fail += int(tm.terminated.sum())
            n_comp += int(tm.time_outs.sum())
            for k in MET:
                sums[k] += cmd.metrics[k].mean().item()
            csum += 1

        tot = n_fail + n_comp
        survival = (n_comp / tot) if tot else float("nan")
        mpbpe = sums["error_body_pos"] / csum * 1000
        mpjpe = sums["error_joint_pos"] / csum
        results[name] = {"survival": round(survival, 4), "e_mpbpe_mm": round(mpbpe, 2),
                         "e_mpjpe_rad": round(mpjpe, 4), "n_steps": csum,
                         "teacher_ckpt": args.teacher_ckpt}
        env.close()
        status = "PASS" if survival >= 0.95 and mpbpe <= 50.0 else "FAIL"
        print(f"  [{status}] {name:40s}  survival={survival:.3f}  mpbpe={mpbpe:.1f}mm")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    n_pass = sum(1 for r in results.values() if r["survival"] >= 0.95 and r["e_mpbpe_mm"] <= 50.0)
    print(f"\n{n_pass}/{len(results)} clips pass quality gate -> {args.out}")
    print("Next: re-export with --quality_json", args.out)


main()
app.close()
