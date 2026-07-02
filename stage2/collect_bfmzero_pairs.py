"""Phase-1 data collector for BFM-Zero -> control-VAE distillation.

Rolls BFM-Zero out on every motion in a pkl and logs, PER CONTROL STEP, the
(state, reference, action, latent) tuple used to distill BFM-Zero's (backward_map, act)
pair into a Gaussian control VAE (see stage2/BFMZERO_DISTILL_PLAN.md):

    s  = proprio actor obs      -> decoder input   (wrapped_env._get_g1env_observation)
    r  = reference/backward obs -> encoder input   (get_backward_observation, per frame)
    a* = model.act(s, z)        -> BC target       (29-D scaled joint-position targets)
    z  = project_z(backward_map(r))                 (BFM-Zero latent, on a sqrt(d)-sphere)

Two knobs matter for beating covariate shift (PLAN sec. 4):
  --dart-noise-std : apply a NOISED action to STEP the env (fatten the visited-state
                     tube) but LABEL each state with the CLEAN teacher action a*. This
                     is DART / noise-injection BC; 0.0 = plain on-policy BC.
  --domain-rand / --obs-noise : DR + observation noise ON during collection so the
                     distilled action is robust (and the state distribution is widened).

Must run in the BFM-Zero env (imports humanoidverse), like
stage3_sim2sim/bfmzero_compare/batch_tracking_inference.py which this mirrors.

Output: <out_dir>/pairs_<mid>.npz with proprio[T,ds], ref[T,dr], action[T,29],
z[T,dz], plus scalar meta. Concatenate across motions for phase-2 training.

Note on the encoder horizon H (PLAN sec. 3): `ref` is logged PER STEP (the full
[T,dr] reference sequence per motion), so the phase-2 trainer can slice any H-frame
window on the fly and sweep H WITHOUT re-collecting. Do not bake H in here --
pre-windowing would fix one H and inflate storage ~Hx. H is a training-time knob.
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


def flatten_obs(obs):
    """dict|tensor obs -> 1-D float32 numpy (single env). Sorted-key concat for dicts."""
    if isinstance(obs, dict):
        parts = [flatten_obs(obs[k]) for k in sorted(obs.keys())]
        return np.concatenate(parts, axis=-1)
    t = obs
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    return t


def main(model_folder: str, data_path: str, out_dir: str, device: str = "cuda",
         simulator: str = "mujoco", max_steps: int = 1200, headless: bool = True,
         start: int = 0, end: int = -1, dart_noise_std: float = 0.0,
         domain_rand: bool = True, obs_noise: bool = True, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_folder, out_dir = Path(model_folder), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_motions = len(joblib.load(data_path))
    end = n_motions if end < 0 else min(end, n_motions)
    print(f"collect: motions [{start},{end}) of {n_motions} -> {out_dir} "
          f"| dart={dart_noise_std} DR={domain_rand} obs_noise={obs_noise}")

    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device).eval()
    config = json.load(open(model_folder / "config.json"))
    use_rh = config["env"].get("root_height_obs", False)
    config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"env.config.headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    # DR / obs-noise are BOTH roles from the plan: robustness + covariate coverage.
    config["env"]["disable_domain_randomization"] = not domain_rand
    config["env"]["disable_obs_noise"] = not obs_noise

    def encode_z(obs):
        z = model.backward_map(obs)
        for s in range(z.shape[0]):
            z[s] = z[s:min(s + 1, z.shape[0])].mean(dim=0)
        return model.project_z(z)

    wrapped_env, _ = HumanoidVerseIsaacConfig(**config["env"]).build(1)
    env = wrapped_env._env
    env_ids = torch.arange(1, dtype=torch.long)
    act_dim = wrapped_env.action_space.shape[-1]

    done = 0
    for mid in range(start, end):
        out = out_dir / f"pairs_{mid}.npz"
        if out.exists():
            done += 1
            continue
        try:
            env.set_is_evaluating(mid)
            _, od = get_backward_observation(env, 0, use_root_height_obs=use_rh)
            obs_bwd, _ = get_backward_observation(env, 0, use_root_height_obs=use_rh)
            z = encode_z(tree_map(lambda x: x[1:], obs_bwd))          # [T-1, dz]
            ref_frames = tree_map(lambda x: x[1:], obs_bwd)           # aligned with z

            # reset env to the motion's initial reference state (mirrors batch_tracking_inference)
            wrapped_env.reset(to_numpy=False)
            root_init = torch.cat([od["ref_body_pos"][0, 0], od["ref_body_rots"][0, 0],
                                   od["ref_body_vels"][0, 0], od["ref_body_angular_vels"][0, 0]])
            dof_init = torch.zeros_like(env.simulator.dof_state.view(1, -1, 2)[0])
            dof_init[..., 0] = od["dof_pos"][0]
            dof_init[..., 1] = od["ref_dof_vel"][0]
            wrapped_env._env.reset_envs_idx(env_ids, target_states={
                "dof_states": dof_init, "root_states": root_init[None]})
            wrapped_env.step(torch.zeros((1, act_dim), dtype=torch.float32), to_numpy=False)
            observation = wrapped_env._get_g1env_observation(to_numpy=False)

            S, Rr, A, Z = [], [], [], []
            ep = min(z.shape[0], max_steps)
            for i in range(ep):
                zi = z[i % len(z)].repeat(1, 1)
                a_star = model.act(observation, zi, mean=True)       # CLEAN teacher action = label

                # log the (s, r, a*, z) tuple at the current (possibly perturbed) state
                S.append(flatten_obs(observation))
                Rr.append(flatten_obs(tree_map(lambda x: x[i], ref_frames)))
                A.append(np.asarray(a_star.detach().cpu().numpy(), np.float32).reshape(-1)[:act_dim])
                Z.append(np.asarray(zi.detach().cpu().numpy(), np.float32).reshape(-1))

                # DART: step with a NOISED action to widen the visited-state tube,
                # but the label above stays the CLEAN a*(s_t).
                a_env = a_star
                if dart_noise_std > 0.0:
                    a_env = a_star + dart_noise_std * torch.randn_like(a_star)
                observation, *_ = wrapped_env.step(a_env, to_numpy=False)

            np.savez(out,
                     proprio=np.stack(S, 0).astype(np.float32),
                     ref=np.stack(Rr, 0).astype(np.float32),
                     action=np.stack(A, 0).astype(np.float32),
                     z=np.stack(Z, 0).astype(np.float32),
                     dart_noise_std=np.float32(dart_noise_std),
                     domain_rand=np.int8(domain_rand), obs_noise=np.int8(obs_noise))
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{end - start} done (last T={len(S)}, "
                      f"ds={len(S[0])}, dr={len(Rr[0])}, dz={len(Z[0])})")
        except Exception as e:
            print(f"  motion {mid} FAILED: {repr(e)[:140]}")
    print(f"collect done: {done}/{end - start} motions -> {out_dir}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
