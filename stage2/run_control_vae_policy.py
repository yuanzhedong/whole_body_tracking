"""Phase-3: closed-loop (G2) eval of the distilled control VAE in the BFM-Zero MuJoCo env.

Runs, per clip, three policies and scores survival + joint tracking:
  * bfm     : BFM-Zero teacher  a = bfm.act(s, z_bfm)                 (baseline)
  * vae_mu  : our VAE           a = D(E(ref_window[H]).mu, s)         (latent used)
  * vae_zero: our VAE           a = D(0, s)                           (latent ablated)

G2 asks: does vae survive >= 0.9x BFM? And does the latent matter IN THE LOOP
(vae_mu vs vae_zero) -- the closed-loop counterpart to the offline z-ablation finding.

Loads a phase-2 sweep checkpoint (control_vae.pt + normalization.npz). Runs in the
BFM-Zero env (imports humanoidverse). Mirrors collect_bfmzero_pairs.py's env setup.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.utils.helpers import get_backward_observation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae_model import MotionVAE  # noqa: E402

FALL_Z = 0.4  # root height below this = fallen (standing G1 pelvis ~0.75 m)


def flatten_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([flatten_obs(obs[k]) for k in sorted(obs.keys())], axis=-1)
    t = obs.detach().cpu().numpy() if hasattr(obs, "detach") else obs
    return np.asarray(t, np.float32).reshape(-1)


def load_vae(ckpt_dir):
    ck = torch.load(Path(ckpt_dir) / "control_vae.pt", map_location="cpu", weights_only=False)
    a = ck["arch"]
    m = MotionVAE(ref_dim=a["ref_dim"], proprio_dim=a["proprio_dim"], act_dim=a["act_dim"], latent=a["latent"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    nz = np.load(Path(ckpt_dir) / "normalization.npz")
    return m, a["horizon"], {k: nz[k].astype(np.float32) for k in nz.files}


def survival_and_track(exec_q, ref_dof):
    exec_q = np.asarray(exec_q, np.float32)
    root_z = exec_q[:, 2]
    survived = float(root_z.min() > FALL_Z)
    n = min(len(exec_q), len(ref_dof))
    jerr = float(np.rad2deg(np.abs(exec_q[:n, 7:36] - ref_dof[:n])).mean()) if n else float("nan")
    return [survived, round(float((root_z > FALL_Z).mean()), 3), round(jerr, 2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--model-folder", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--out", default="stage2/out/g2_eval")
    p.add_argument("--num-clips", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vae, H, norm = load_vae(args.ckpt_dir)
    print(f"VAE: H={H} latent={vae.latent} ref_dim={vae.ref_dim} | ckpt={args.ckpt_dir}")

    @torch.no_grad()
    def vae_action(Rfull, t, proprio, use_latent):
        idx = np.minimum(np.arange(t, t + H), len(Rfull) - 1)
        w = ((Rfull[idx].reshape(-1) - norm["ref_mu"]) / norm["ref_sd"]).astype(np.float32)
        s = ((proprio - norm["pro_mu"]) / norm["pro_sd"]).astype(np.float32)
        mu, _ = vae.encode(torch.tensor(w)[None]) if use_latent else (torch.zeros(1, vae.latent), None)
        a = vae.decode(mu, torch.tensor(s)[None]).numpy()[0]
        return (a * norm["act_sd"] + norm["act_mu"]).astype(np.float32)

    model = load_model_from_checkpoint_dir(Path(args.model_folder) / "checkpoint", device=args.device)
    model.to(args.device).eval()
    config = json.load(open(Path(args.model_folder) / "config.json"))
    use_rh = config["env"].get("root_height_obs", False)
    config["env"]["lafan_tail_path"] = str(Path(args.data_path).resolve())
    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append("env.config.headless=True")
    config["env"]["hydra_overrides"].append("simulator=mujoco")
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
    n_motions = min(args.num_clips, len(joblib.load(args.data_path)))

    def rollout(mid, policy):
        """policy(i, observation, z_bfm_i, Rfull) -> action tensor [1, act_dim]."""
        env.set_is_evaluating(mid)
        _, od = get_backward_observation(env, 0, use_root_height_obs=use_rh)
        obs_bwd, _ = get_backward_observation(env, 0, use_root_height_obs=use_rh)
        z_bfm = encode_z(tree_map(lambda x: x[1:], obs_bwd))
        ref_frames = tree_map(lambda x: x[1:], obs_bwd)
        T = z_bfm.shape[0]
        Rfull = np.stack([flatten_obs(tree_map(lambda x: x[t], ref_frames)) for t in range(T)])
        ref_dof = od["dof_pos"].cpu().numpy()

        wrapped_env.reset(to_numpy=False)
        root_init = torch.cat([od["ref_body_pos"][0, 0], od["ref_body_rots"][0, 0],
                               od["ref_body_vels"][0, 0], od["ref_body_angular_vels"][0, 0]])
        dof_init = torch.zeros_like(env.simulator.dof_state.view(1, -1, 2)[0])
        dof_init[..., 0] = od["dof_pos"][0]; dof_init[..., 1] = od["ref_dof_vel"][0]
        wrapped_env._env.reset_envs_idx(env_ids, target_states={"dof_states": dof_init, "root_states": root_init[None]})
        wrapped_env.step(torch.zeros((1, act_dim), dtype=torch.float32), to_numpy=False)
        observation = wrapped_env._get_g1env_observation(to_numpy=False)

        exec_q = [np.asarray(wrapped_env._get_qpos_qvel(to_numpy=True)[0]).reshape(-1)[:36].copy()]
        for i in range(min(T, args.max_steps)):
            a = policy(i, observation, z_bfm[i % T], Rfull)
            observation, *_ = wrapped_env.step(a, to_numpy=False)
            exec_q.append(np.asarray(wrapped_env._get_qpos_qvel(to_numpy=True)[0]).reshape(-1)[:36].copy())
        return np.stack(exec_q, 0), ref_dof

    def pol_bfm(i, obs, zb, Rfull):
        return model.act(obs, zb.repeat(1, 1), mean=True)

    def make_vae_pol(use_latent):
        def pol(i, obs, zb, Rfull):
            return torch.tensor(vae_action(Rfull, i, flatten_obs(obs), use_latent), dtype=torch.float32)[None]
        return pol

    policies = {"bfm": pol_bfm, "vae_mu": make_vae_pol(True), "vae_zero": make_vae_pol(False)}
    results = []
    for mid in range(n_motions):
        row = {"clip": mid}
        try:
            for name, pol in policies.items():
                eq, rdof = rollout(mid, pol)
                row[name] = survival_and_track(eq, rdof)
            print(f"clip {mid}: bfm surv={row['bfm'][0]:.0f} jerr={row['bfm'][2]:.1f} | "
                  f"vae_mu surv={row['vae_mu'][0]:.0f} jerr={row['vae_mu'][2]:.1f} | "
                  f"vae_zero surv={row['vae_zero'][0]:.0f} jerr={row['vae_zero'][2]:.1f}")
        except Exception as e:
            row["error"] = repr(e)[:160]; print(f"clip {mid} FAILED: {row['error']}")
        results.append(row)

    def col(name, idx):
        v = [r[name][idx] for r in results if isinstance(r.get(name), list)]
        return round(float(np.mean(v)), 3) if v else None
    agg = {"n": len([r for r in results if "bfm" in r]),
           "survival_bfm": col("bfm", 0), "survival_vae_mu": col("vae_mu", 0), "survival_vae_zero": col("vae_zero", 0),
           "jerr_bfm": col("bfm", 2), "jerr_vae_mu": col("vae_mu", 2), "jerr_vae_zero": col("vae_zero", 2)}
    json.dump({"agg": agg, "per_clip": results, "ckpt": args.ckpt_dir}, open(out / "g2_results.json", "w"), indent=2)
    print("\n=== G2 aggregate ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
