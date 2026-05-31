"""Deterministic tracking evaluation for a trained policy.

Computes the metrics BeyondMimic defines (paper arXiv:2508.08241):
  - success_rate : fraction of episodes that complete the motion (time_out) without
                   a failure termination (the robot falling / drifting off-reference).
  - E_mpbpe      : mean body-part position error relative to the anchor/root (m)  -> error_body_pos
  - E_g_anchor   : global anchor (root) position error (m)                        -> error_anchor_pos
  - E_mpjpe      : mean joint position error (rad)                                -> error_joint_pos
  - plus body orientation / velocity errors for completeness.

Runs many envs in parallel with the *deterministic* inference policy (no exploration
noise) over several motion lengths, so the averages are stable. Runs on the working
Isaac Sim 4.5 / Isaac Lab 2.1 stack, headless, on a 4090 (the .venv torch has no
Blackwell kernels -> pin CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<a 4090>).
"""
import argparse, sys, json
sys.path.append("/ws/user/yzdong/src/github/whole_body_tracking/scripts/rsl_rl")
from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--motion_file", required=True)
parser.add_argument("--episodes", type=float, default=5.0, help="how many motion-lengths to run")
parser.add_argument("--warmup", type=int, default=20, help="steps to skip after each reset before counting errors")
parser.add_argument("--out", default="/tmp/eval_metrics.json")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
# Force headless: this box's RTX renderer segfaults at window creation (driver 595),
# and eval never needs a viewport. Avoids the "Failed to acquire IWindowing" crash.
args.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import whole_body_tracking.tasks  # noqa: F401,E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

METRIC_KEYS = ["error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot",
               "error_joint_pos", "error_joint_vel", "error_body_lin_vel", "error_body_ang_vel"]


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.motion.motion_file = args.motion_file
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    try:
        cmd = uenv.command_manager.get_term("motion")
    except Exception:
        cmd = uenv.command_manager._terms["motion"]
    tm = uenv.termination_manager

    motion_len = int(cmd.motion.time_step_total)
    total_steps = int(motion_len * args.episodes)

    # accumulators (sum of per-env mean metric over counted steps)
    sums = {k: 0.0 for k in METRIC_KEYS}
    counted = 0
    n_fail = 0      # failure terminations (robot fell / drifted off-reference)
    n_complete = 0  # time_outs (reached motion end = completed the clip)
    # steps since last reset, per env, to skip warmup transients in the error average
    since_reset = torch.zeros(uenv.num_envs, dtype=torch.long, device=uenv.device)

    obs, _ = env.get_observations()
    for t in range(total_steps):
        with torch.inference_mode():
            act = policy(obs)
            obs, _, dones, _ = env.step(act)

        terminated = tm.terminated         # failures (excludes time_out)
        time_outs = tm.time_outs           # truncation = motion completed
        n_fail += int(terminated.sum().item())
        n_complete += int(time_outs.sum().item())

        # accumulate tracking errors only on envs that are past warmup (steady tracking)
        mask = since_reset >= args.warmup
        if mask.any():
            m = mask.float()
            denom = m.sum().item()
            for k in METRIC_KEYS:
                sums[k] += (cmd.metrics[k] * m).sum().item() / denom
            counted += 1

        since_reset += 1
        since_reset[dones.bool()] = 0

    total_eps = n_complete + n_fail
    success_rate = (n_complete / total_eps) if total_eps > 0 else float("nan")
    means = {k: (sums[k] / counted if counted else float("nan")) for k in METRIC_KEYS}

    results = {
        "ckpt": args.ckpt,
        "motion_file": args.motion_file,
        "num_envs": args.num_envs,
        "motion_len_steps": motion_len,
        "total_env_steps": total_steps * args.num_envs,
        "episodes_observed": total_eps,
        "n_complete_timeout": n_complete,
        "n_fail": n_fail,
        "success_rate": success_rate,                       # paper: completes motion w/o falling
        "E_mpbpe_m": means["error_body_pos"],               # root-relative body pos err (m)
        "E_mpbpe_mm": means["error_body_pos"] * 1000.0,
        "E_g_anchor_pos_m": means["error_anchor_pos"],      # global root/anchor pos err (m)
        "E_mpjpe_rad": means["error_joint_pos"],            # joint pos err (rad)
        "E_body_rot_rad": means["error_body_rot"],
        "E_body_lin_vel": means["error_body_lin_vel"],
        "E_body_ang_vel": means["error_body_ang_vel"],
        "raw_means": means,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    # write to stderr too (Kit hijacks stdout after SimulationApp starts)
    sys.stderr.write("EVAL_RESULTS " + json.dumps(results) + "\n")
    sys.stderr.flush()
    print("EVAL_DONE", json.dumps(results))
    env.close()


main()
simulation_app.close()
