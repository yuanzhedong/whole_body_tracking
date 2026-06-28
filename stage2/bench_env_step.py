"""Micro-benchmark: ms/step of the tracking env at several num_envs in ONE Isaac session.
Answers 'why is eval slow' empirically — is per-step time fixed-overhead-bound or compute-bound?

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
    .venv/bin/python stage2/bench_env_step.py --env_counts 128 512 2048 --steps 60
"""
import argparse, os, sys, time
sys.path.insert(0, "/ws/user/yzdong/src/github/whole_body_tracking/scripts/rsl_rl")
from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-G1-v0")
parser.add_argument("--env_counts", type=int, nargs="+", default=[128, 512, 2048])
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--teacher_ckpt",
                    default="/ws/user/yzdong/src/github/whole_body_tracking/logs/rsl_rl/g1_flat/2026-05-30_15-59-30_walk_4090_full30k/model_29999.pt")
parser.add_argument("--motion_file",
                    default="/ws/user/yzdong/src/github/whole_body_tracking/artifacts/walk1_subject1_0_33:v0/motion.npz")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import torch
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import whole_body_tracking.tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    env_cfg.commands.motion.motion_file = args.motion_file

    print("\n=== ENV-STEP BENCHMARK (ms/step, with per-step .item() sync as in the real eval) ===")
    print(f"{'num_envs':>9} | {'ms/step':>8} | {'ms/step no-sync':>15} | {'GPU work/step est':>17}")
    for ne in args.env_counts:
        env_cfg.scene.num_envs = ne
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env)
        uenv = env.unwrapped
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.teacher_ckpt)
        policy = runner.get_inference_policy(device=uenv.device)
        cmd = uenv.command_manager.get_term("motion")
        tm = uenv.termination_manager
        MET = ["error_body_pos", "error_joint_pos"]

        obs, _ = env.get_observations()
        # warmup
        for _ in range(8):
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))

        # timed WITH per-step CPU sync (mirrors the real eval loop)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.steps):
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            _ = int(tm.terminated.sum()); _ = int(tm.time_outs.sum())
            for k in MET: _ = cmd.metrics[k].mean().item()
        torch.cuda.synchronize()
        ms_sync = (time.time() - t0) / args.steps * 1000

        # timed WITHOUT per-step sync (sync once at end)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.steps):
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
        torch.cuda.synchronize()
        ms_nosync = (time.time() - t0) / args.steps * 1000

        print(f"{ne:>9} | {ms_sync:>8.1f} | {ms_nosync:>15.1f} | {'(see nosync)':>17}")
        env.close()

    print("\nReading: if ms/step is ~flat across env counts -> fixed-overhead/sync bound (more envs = free samples).")
    print("If ms/step rises with env count -> physics-compute bound (more envs genuinely costs more).")
    print("Compare sync vs no-sync columns to see how much the per-step .item() GPU->CPU sync costs.")


if __name__ == "__main__":
    main()
    app.close()
