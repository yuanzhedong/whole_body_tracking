"""Does the eval slow down on DECODED motion because the robot falls (resets)?
Runs the real eval loop on original vs decoded motion, reporting ms/step AND
termination/timeout counts AND reset rate. Unbuffered.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
    .venv/bin/python -u stage2/bench_reset.py --steps 300 --num_envs 128
"""
import argparse, os, sys, time
sys.path.insert(0, "/ws/user/yzdong/src/github/whole_body_tracking/scripts/rsl_rl")
from isaaclab.app import AppLauncher
import cli_args


def log(*a): print(*a, flush=True)


WBT = "/ws/user/yzdong/src/github/whole_body_tracking"
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--teacher_ckpt",
                    default=f"{WBT}/logs/rsl_rl/g1_flat/2026-05-30_15-59-30_walk_4090_full30k/model_29999.pt")
parser.add_argument("--orig", default=f"{WBT}/artifacts/walk1_subject1_0_33:v0/motion.npz")
parser.add_argument("--decoded", default=f"{WBT}/stage2/out/sim2sim_et_decoded/walk1_subject1_0_33_decoded.npz")
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
    env_cfg.scene.num_envs = args.num_envs
    N = args.steps

    log(f"\n=== RESET-COST BENCHMARK ({N} steps, {args.num_envs} envs) ===")
    log(f"{'motion':>10} | {'ms/step':>8} | {'resets/step':>11} | {'term/step':>9} | "
        f"{'timeout/step':>12} | {'total_resets':>12}")

    for label, path in [("ORIGINAL", args.orig), ("DECODED", args.decoded)]:
        if not os.path.exists(path):
            log(f"{label:>10} | MISSING: {path}")
            continue
        env_cfg.commands.motion.motion_file = path
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env)
        uenv = env.unwrapped
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.teacher_ckpt)
        policy = runner.get_inference_policy(device=uenv.device)
        tm = uenv.termination_manager

        obs, _ = env.get_observations()
        for _ in range(10):
            with torch.inference_mode(): obs, _, _, _ = env.step(policy(obs))

        torch.cuda.synchronize(); t0 = time.time()
        n_term = n_to = 0
        for _ in range(N):
            with torch.inference_mode(): obs, _, _, _ = env.step(policy(obs))
            n_term += int(tm.terminated.sum())
            n_to   += int(tm.time_outs.sum())
        torch.cuda.synchronize()
        dt = time.time() - t0
        total_resets = n_term + n_to
        log(f"{label:>10} | {dt/N*1000:>8.1f} | {total_resets/N:>11.1f} | {n_term/N:>9.1f} | "
            f"{n_to/N:>12.1f} | {total_resets:>12}")
        env.close()

    log("\n=== READ ===")
    log("If DECODED ms/step >> ORIGINAL ms/step AND term/step is high -> reset-on-fall is the cost.")
    log("If both similar ms/step -> resets are NOT the cause; look elsewhere.")
    log("resets/step ~ num_envs means EVERY env resets EVERY step (worst case).")


if __name__ == "__main__":
    main()
    app.close()
