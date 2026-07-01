"""Track A step (3): full-schema BFM-Zero state-action collection (DATA_PIPELINE.md §6).

Rolls BFM-Zero out on each motion in a gt pkl and logs the complete per-step record:
  action(T,29) z(T,256) qpos(T,36) qvel(T,35) obs_state(T,64) obs_privileged(T,P)
  last_action(T,29) ref_dof_pos(T,29) ref_dof_vel(T,29) ref_body_pos(T,B,3)
  ref_body_rots(T,B,4) alive(T,) reward(T,) ref_frame_idx(T,)

Two modes (DATA_PIPELINE.md §4 A3):
  * onpolicy       — closed-loop: log states BFM-Zero actually visits + the action taken.
  * teacher_forced — reset the robot onto the reference each control step; log (ref state, action).

Output: <out>/<mode>/traj_<idx>_<clip>.npz. Then run pack_state_action.py for
normalization.npz + manifest.json + dataset_stats.json. Run in the BFM-Zero env (.venv-bfm).
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.utils.helpers import get_backward_observation

FALL_Z = 0.4


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main(model_folder: str, data_path: str, out_dir: str, mode: str = "onpolicy",
         device: str = "cuda", simulator: str = "mujoco", max_steps: int = 2000,
         start: int = 0, end: int = -1):
    assert mode in ("onpolicy", "teacher_forced")
    out = Path(out_dir) / mode
    out.mkdir(parents=True, exist_ok=True)
    motions = joblib.load(data_path)
    names = list(motions.keys())
    n = len(names)
    end = n if end < 0 else min(end, n)
    print(f"collect_state_action [{mode}] motions [{start},{end}) of {n} -> {out}")

    model = load_model_from_checkpoint_dir(Path(model_folder) / "checkpoint", device=device)
    model.to(device).eval()
    config = json.load(open(Path(model_folder) / "config.json"))
    use_rh = config["env"].get("root_height_obs", False)
    config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append("env.config.headless=True")
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
    act_dim = wrapped_env.action_space.shape[-1]

    def reset_to_ref(od, i):
        root = torch.cat([od["ref_body_pos"][i, 0], od["ref_body_rots"][i, 0],
                          od["ref_body_vels"][i, 0], od["ref_body_angular_vels"][i, 0]])
        dof = torch.zeros_like(env.simulator.dof_state.view(1, -1, 2)[0])
        dof[..., 0] = od["dof_pos"][i]; dof[..., 1] = od["ref_dof_vel"][i]
        wrapped_env._env.reset_envs_idx(env_ids, target_states={"dof_states": dof, "root_states": root[None]})

    done = 0
    for mid in range(start, end):
        clip = names[mid]
        f = out / f"traj_{mid}_{clip[:40]}.npz"
        if f.exists():
            done += 1; continue
        try:
            env.set_is_evaluating(mid)
            _, od = get_backward_observation(env, 0, use_root_height_obs=use_rh)
            obs_bwd, _ = get_backward_observation(env, 0, use_root_height_obs=use_rh)
            z = encode_z(tree_map(lambda x: x[1:], obs_bwd))
            T = min(z.shape[0], max_steps)

            wrapped_env.reset(to_numpy=False)
            reset_to_ref(od, 0)
            wrapped_env.step(torch.zeros((1, act_dim), dtype=torch.float32), to_numpy=False)
            obs = wrapped_env._get_g1env_observation(to_numpy=False)

            rec = {k: [] for k in ("action", "z", "qpos", "qvel", "obs_state", "obs_privileged",
                                   "last_action", "ref_dof_pos", "ref_dof_vel", "ref_body_pos",
                                   "ref_body_rots", "alive", "reward", "ref_frame_idx")}
            for i in range(T):
                if mode == "teacher_forced":
                    reset_to_ref(od, i)
                    obs = wrapped_env._get_g1env_observation(to_numpy=False)
                a = model.act(obs, z[i % z.shape[0]].repeat(1, 1), mean=True)
                qpos, qvel = wrapped_env._get_qpos_qvel(to_numpy=True)
                rec["action"].append(to_np(a).reshape(-1)[:act_dim])
                rec["z"].append(to_np(z[i % z.shape[0]]).reshape(-1))
                rec["qpos"].append(np.asarray(qpos).reshape(-1)[:36])
                rec["qvel"].append(np.asarray(qvel).reshape(-1)[:35])
                rec["obs_state"].append(to_np(obs["state"]).reshape(-1))
                rec["obs_privileged"].append(to_np(obs["privileged_state"]).reshape(-1))
                rec["last_action"].append(to_np(obs.get("last_action", a)).reshape(-1)[:act_dim])
                rec["ref_dof_pos"].append(to_np(od["dof_pos"][i]).reshape(-1))
                rec["ref_dof_vel"].append(to_np(od["ref_dof_vel"][i]).reshape(-1))
                rec["ref_body_pos"].append(to_np(od["ref_body_pos"][i]))
                rec["ref_body_rots"].append(to_np(od["ref_body_rots"][i]))
                rec["ref_frame_idx"].append(i)
                obs, reward, terminated, truncated, info = wrapped_env.step(a, to_numpy=False)
                zpos = float(np.asarray(wrapped_env._get_qpos_qvel(to_numpy=True)[0]).reshape(-1)[2])
                rec["alive"].append(1.0 if zpos > FALL_Z else 0.0)
                rec["reward"].append(float(to_np(reward).reshape(-1)[0]))
                if mode == "onpolicy" and zpos < FALL_Z:
                    break
            arrs = {k: np.asarray(v, np.float32) for k, v in rec.items()}
            arrs["clip"] = clip
            np.savez(f, **arrs)
            done += 1
            if done % 20 == 0:
                print(f"  {done} clips ({mode}), last T={len(arrs['action'])}, "
                      f"survival~{float(arrs['alive'].mean()):.2f}")
        except Exception as e:
            print(f"  motion {mid} FAILED: {repr(e)[:120]}")
    print(f"done: {done} trajectories -> {out}")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
