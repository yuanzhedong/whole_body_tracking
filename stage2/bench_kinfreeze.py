"""Quick validation of the freeze idea BEFORE implementing it properly.
Make all robots fall (zero actions -> collapse), then compare per-step time:
  (A) fallen, on the ground, dynamic         -> the expensive case
  (B) fallen, lifted into the air (no ground contact, once)  -> proxy for 'freeze removes contacts'
If (B) << (A), contacts are the cost and a real kinematic/collision-disable freeze will be fast.

  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u stage2/bench_kinfreeze.py
"""
import os, sys, time
sys.path.insert(0, "/ws/user/yzdong/src/github/whole_body_tracking/scripts/rsl_rl")
from isaaclab.app import AppLauncher
import cli_args

def log(*a): print(*a, flush=True)

WBT = "/ws/user/yzdong/src/github/whole_body_tracking"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--motion_file", default=f"{WBT}/artifacts/walk1_subject1_0_33:v0/motion.npz")
parser.add_argument("--collapse_steps", type=int, default=150)
parser.add_argument("--measure_steps", type=int, default=40)
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

def timeit(env, action, n):
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        with torch.inference_mode(): env.step(action)
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000

@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.motion.motion_file = args.motion_file
    # don't let terminations reset robots during this test
    env_cfg.terminations.anchor_pos = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.ee_body_pos = None
    env_cfg.terminations.time_out = None
    env_cfg.episode_length_s = 1.0e6
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env); uenv = env.unwrapped
    robot = uenv.scene["robot"]
    nact = env.num_actions if hasattr(env, "num_actions") else uenv.action_manager.total_action_dim
    zero = torch.zeros((args.num_envs, nact), device=uenv.device)

    env.get_observations()
    log(f"\n=== FREEZE-IDEA VALIDATION ({args.num_envs} envs) ===")

    # warmup upright
    ms_upright = timeit(env, zero, args.measure_steps)
    log(f"  (A0) upright/early  : {ms_upright:6.1f} ms/step")

    # collapse: zero joint targets -> robots go limp and fall
    log(f"  collapsing robots ({args.collapse_steps} steps of zero action)...")
    for _ in range(args.collapse_steps):
        with torch.inference_mode(): env.step(zero)

    ms_fallen = timeit(env, zero, args.measure_steps)
    log(f"  (A)  fallen ON GROUND (dynamic, contacts): {ms_fallen:6.1f} ms/step")

    # lift all robots high once (remove ground contact) — proxy for 'freeze removes contacts'
    root = robot.data.root_state_w.clone()
    root[:, 2] += 50.0       # +50m
    root[:, 7:13] = 0.0      # zero velocity
    robot.write_root_state_to_sim(root)
    # measure WITHOUT re-lifting (they free-fall slowly; stay high & contact-free for ~40 steps)
    ms_air = timeit(env, zero, args.measure_steps)
    log(f"  (B)  fallen IN AIR   (no ground contact): {ms_air:6.1f} ms/step")

    log("\n=== READ ===")
    log(f"  ground-contact overhead = (A) - (B) = {ms_fallen - ms_air:6.1f} ms/step")
    if ms_air < ms_fallen * 0.6:
        log("  -> contacts ARE the cost. A real freeze (kinematic / disable collisions on FAILED")
        log("     envs) would make fallen robots ~as cheap as upright. The idea works.")
    else:
        log("  -> lifting didn't help much; self-collisions (not ground) may dominate -> a full")
        log("     collision-disable (not just lift) is needed. Informs the implementation.")
    env.close()

if __name__ == "__main__":
    main()
    app.close()
