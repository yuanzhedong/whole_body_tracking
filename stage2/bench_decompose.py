"""Root-cause decomposition of Phase-2 eval per-step cost. Unbuffered, flushes each line.

Isolates, at several env counts, the wall-clock of:
  (a) policy(obs)                      — NN inference
  (b) env.step(a)                      — full sim step, GPU-synced (true cost)
  (c) per-step readout                 — int(terminated.sum()), metrics.item()  [the eager sync]
  (d) full eval-style step             — (a)+(b)+(c) as the real loop does it
  (e) reset cost                       — fraction of steps that triggered resets

Run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
    .venv/bin/python -u stage2/bench_decompose.py --env_counts 128 1024 4096 --steps 80
"""
import argparse, os, sys, time
sys.path.insert(0, "/ws/user/yzdong/src/github/whole_body_tracking/scripts/rsl_rl")
from isaaclab.app import AppLauncher
import cli_args


def log(*a):
    print(*a, flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-G1-v0")
parser.add_argument("--env_counts", type=int, nargs="+", default=[128, 1024, 4096])
parser.add_argument("--steps", type=int, default=80)
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


def timeit(fn, n, sync=True):
    if sync:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0  # ms/call


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    env_cfg.commands.motion.motion_file = args.motion_file
    MET = ["error_body_pos", "error_joint_pos"]
    N = args.steps

    log(f"\n=== PER-STEP COST DECOMPOSITION (ms/step) — {N} steps each, walk_0_33 ===")
    log(f"{'envs':>6} | {'policy':>7} | {'step(synced)':>12} | {'step(no-sync)':>13} | "
        f"{'readout(.item)':>14} | {'full-eval-loop':>14} | {'GPU%@step':>9}")
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

        obs, _ = env.get_observations()
        for _ in range(10):  # warmup
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))

        # (a) policy inference only
        a_buf = [None]
        def _pol():
            with torch.inference_mode():
                a_buf[0] = policy(obs)
        t_pol = timeit(_pol, N)

        # (b) full step, GPU-synced each call (true per-step cost incl. CPU-side python)
        def _step_sync():
            with torch.inference_mode():
                env.step(policy(obs))
        t_step_sync = timeit(_step_sync, N, sync=True)

        # (c) full step WITHOUT per-call sync (sync once at end) — exposes pipelining headroom
        def _step_nosync():
            with torch.inference_mode():
                env.step(policy(obs))
        t_step_nosync = timeit(_step_nosync, N, sync=False)  # outer sync only at end via timeit

        # (d) the eager per-step readout cost in isolation (on current obs/tensors)
        def _readout():
            _ = int(tm.terminated.sum()); _ = int(tm.time_outs.sum())
            for k in MET: _ = cmd.metrics[k].mean().item()
        t_readout = timeit(_readout, N, sync=False)  # .item() itself forces sync internally

        # (e) full eval-style loop (step + eager readout) — what the real eval does
        def _full():
            with torch.inference_mode():
                env.step(policy(obs))
            _ = int(tm.terminated.sum()); _ = int(tm.time_outs.sum())
            for k in MET: _ = cmd.metrics[k].mean().item()
        t_full = timeit(_full, N, sync=False)

        # crude GPU util sample during stepping
        gpu = "n/a"
        try:
            import subprocess
            gpu = subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits","-i","1"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            pass

        log(f"{ne:>6} | {t_pol:>7.2f} | {t_step_sync:>12.2f} | {t_step_nosync:>13.2f} | "
            f"{t_readout:>14.2f} | {t_full:>14.2f} | {gpu:>9}")
        env.close()

    log("\n=== HOW TO READ ===")
    log("- step(synced) flat across envs  -> overhead-bound (more envs ~free). Rises -> compute-bound.")
    log("- step(no-sync) << step(synced)  -> the GPU pipeline has slack; per-call sync is wasting it.")
    log("- full-eval-loop >> step(synced) -> the eager readout (.item) is the dominant extra cost.")
    log("- readout(.item) column = isolated cost of the per-step GPU->CPU sync.")


if __name__ == "__main__":
    main()
    app.close()
