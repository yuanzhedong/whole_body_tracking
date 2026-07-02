"""Batched BFM-Zero tracking: build the env ONCE, roll out every motion in the pkl.

Saves executed/reference qpos_36 per motion to <out_dir>/rollout_<i>.npz. Much faster
than tracking_inference.py for many clips (no per-process env build). MuJoCo only.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

import json
from pathlib import Path
import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.utils.helpers import get_backward_observation
import humanoidverse

HV = Path(humanoidverse.__file__).parent


def main(model_folder: Path, data_path: Path, out_dir: Path, device: str = "cuda",
         simulator: str = "mujoco", max_steps: int = 1200, headless: bool = True,
         start: int = 0, end: int = -1):
    model_folder, out_dir = Path(model_folder), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_motions = len(joblib.load(data_path))
    end = n_motions if end < 0 else min(end, n_motions)
    print(f"batch: motions [{start},{end}) of {n_motions} -> {out_dir}")

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device).eval()
    config = json.load(open(model_folder / "config.json"))
    use_rh = config["env"].get("root_height_obs", False)
    config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = False
    config["env"]["disable_obs_noise"] = False

    def encode_z(obs):
        z = model.backward_map(obs)
        for s in range(z.shape[0]):
            z[s] = z[s:min(s + 1, z.shape[0])].mean(dim=0)
        return model.project_z(z)

    wrapped_env, _ = HumanoidVerseIsaacConfig(**config["env"]).build(1)
    env = wrapped_env._env
    env_ids = torch.arange(1, dtype=torch.long)

    done = 0
    for mid in range(start, end):
        out = out_dir / f"rollout_{mid}.npz"
        if out.exists():
            done += 1
            continue
        try:
            env.set_is_evaluating(mid)
            obs, od = get_backward_observation(env, 0, use_root_height_obs=use_rh)
            expert_qpos = np.concatenate([
                od["ref_body_pos"][:, 0].cpu().numpy(),
                np.roll(od["ref_body_rots"][:, 0].cpu().numpy(), 1, axis=-1),
                od["dof_pos"].cpu().numpy()], axis=-1)
            z = encode_z(tree_map(lambda x: x[1:], obs))

            wrapped_env.reset(to_numpy=False)
            root_init = torch.cat([od["ref_body_pos"][0, 0], od["ref_body_rots"][0, 0],
                                   od["ref_body_vels"][0, 0], od["ref_body_angular_vels"][0, 0]])
            dof_init = torch.zeros_like(env.simulator.dof_state.view(1, -1, 2)[0])
            dof_init[..., 0] = od["dof_pos"][0]
            dof_init[..., 1] = od["ref_dof_vel"][0]
            wrapped_env._env.reset_envs_idx(env_ids, target_states={
                "dof_states": dof_init, "root_states": root_init[None]})
            wrapped_env.step(torch.zeros((1, wrapped_env.action_space.shape[-1]), dtype=torch.float32), to_numpy=False)
            observation = wrapped_env._get_g1env_observation(to_numpy=False)

            exec_q = [np.asarray(wrapped_env._get_qpos_qvel(to_numpy=True)[0]).reshape(-1)[:36].copy()]
            ep = min(z.shape[0], max_steps)
            for i in range(ep):
                action = model.act(observation, z[i % len(z)].repeat(1, 1), mean=True)
                observation, *_ = wrapped_env.step(action, to_numpy=False)
                exec_q.append(np.asarray(wrapped_env._get_qpos_qvel(to_numpy=True)[0]).reshape(-1)[:36].copy())
            ex = np.stack(exec_q, 0).astype(np.float32)
            ref = np.asarray(expert_qpos, dtype=np.float32)[:len(ex)]
            np.savez(out, executed_qpos_36=ex, reference_qpos_36=ref)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{n_motions} done")
        except Exception as e:
            print(f"  motion {mid} FAILED: {repr(e)[:120]}")
    print(f"batch done: {done}/{n_motions}")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
