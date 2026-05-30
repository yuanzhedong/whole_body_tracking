"""Headless rollout of a trained tracking policy; log per-frame robot states.
Runs on the working Isaac Sim 4.5 / Isaac Lab 2.1 stack (no rendering)."""
import argparse, sys
sys.path.append("/ws/user/yzdong/src/github/whole_body_tracking/scripts/rsl_rl")
from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--motion_file", required=True)
parser.add_argument("--out", default="/tmp/rollout_states.npz")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import whole_body_tracking.tasks  # noqa: F401,E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


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

    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)
    origin = env.unwrapped.scene.env_origins[0].clone()  # subtract so root is world-centered

    # Log per-body world poses (forward kinematics from the sim) in addition to root+joints.
    # The renderer replays these link transforms directly, so the rendered mesh matches the
    # policy exactly without needing physics to propagate joint state to the render.
    log = {"root_pos": [], "root_quat": [], "joint_pos": [],
           "body_pos": [], "body_quat": [], "resets": []}
    obs, _ = env.get_observations()
    for i in range(args.steps):
        with torch.inference_mode():
            act = policy(obs)
            obs, _, dones, _ = env.step(act)
        rp = (robot.data.root_pos_w[0] - origin).cpu().numpy().copy()
        log["root_pos"].append(rp)
        log["root_quat"].append(robot.data.root_quat_w[0].cpu().numpy().copy())  # wxyz
        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        # body_pos_w: (num_bodies, 3), body_quat_w: (num_bodies, 4) wxyz
        bp = (robot.data.body_pos_w[0] - origin).cpu().numpy().copy()
        log["body_pos"].append(bp)
        log["body_quat"].append(robot.data.body_quat_w[0].cpu().numpy().copy())  # wxyz
        log["resets"].append(int(dones[0].item()) if hasattr(dones, "__getitem__") else 0)

    out = {
        "root_pos": np.stack(log["root_pos"]),
        "root_quat": np.stack(log["root_quat"]),
        "joint_pos": np.stack(log["joint_pos"]),
        "body_pos": np.stack(log["body_pos"]),    # (T, num_bodies, 3) world, origin-subtracted
        "body_quat": np.stack(log["body_quat"]),  # (T, num_bodies, 4) wxyz
        "resets": np.array(log["resets"]),
        "joint_names": np.array(joint_names),
        "body_names": np.array(body_names),
        "fps": np.array([50]),
    }
    np.savez(args.out, **out)
    print("ROLLOUT_SAVED", args.out, "joint_pos", out["joint_pos"].shape,
          "body_pos", out["body_pos"].shape, "resets", int(out["resets"].sum()))
    env.close()


main()
simulation_app.close()
