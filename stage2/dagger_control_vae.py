"""Phase-4: DAgger distillation of BFM-Zero into the control VAE.

Fixes the covariate shift that sinks pure offline BC (EXPERIMENTS_bfmzero_distill.md,
Finding #2: BC survival 0.5 vs BFM 1.0). Each iteration:

  1. AGGREGATE: roll out the CURRENT student  a = D(E(ref_window).mu, s)  in the BFM-Zero
     MuJoCo env; at every student-visited state s_t, query the TEACHER label
        a*_t = BFM.act(s_t, z_bfm_t)          (z_bfm = project_z(backward_map(ref)))
     and log (s_t, r_t, a*_t). The student rollout doubles as the survival eval.
  2. TRAIN: retrain the VAE (recon + beta*KL) on the AGGREGATED buffer (BC data + all
     DAgger data), with normalization frozen from the warm-start checkpoint.

Warm-starts from a phase-2 sweep checkpoint (control_vae.pt + normalization.npz). Runs in
the BFM-Zero env (.venv-bfm). Prints the per-iteration survival curve.
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
from vae_model import MotionVAE, kl_divergence  # noqa: E402

FALL_Z = 0.4


def flatten_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([flatten_obs(obs[k]) for k in sorted(obs.keys())], axis=-1)
    t = obs.detach().cpu().numpy() if hasattr(obs, "detach") else obs
    return np.asarray(t, np.float32).reshape(-1)


def load_warm(ckpt_dir):
    ck = torch.load(Path(ckpt_dir) / "control_vae.pt", map_location="cpu", weights_only=False)
    a = ck["arch"]
    m = MotionVAE(ref_dim=a["ref_dim"], proprio_dim=a["proprio_dim"], act_dim=a["act_dim"], latent=a["latent"])
    m.load_state_dict(ck["state_dict"])
    nz = np.load(Path(ckpt_dir) / "normalization.npz")
    return m, a["horizon"], {k: nz[k].astype(np.float32) for k in nz.files}


def build_windows(buffer, H):
    """buffer: list of {s:[T,ds], r:[T,dr], a:[T,da]} -> windowed arrays for training."""
    Xr, Xp, Ya = [], [], []
    for m in buffer:
        r, s, a = m["r"], m["s"], m["a"]
        T = len(r)
        for t in range(T):
            idx = np.minimum(np.arange(t, t + H), T - 1)
            Xr.append(r[idx].reshape(-1)); Xp.append(s[t]); Ya.append(a[t])
    return np.stack(Xr).astype(np.float32), np.stack(Xp).astype(np.float32), np.stack(Ya).astype(np.float32)


def train_on_buffer(vae, buffer, norm, H, epochs, dev, beta=0.01, bs=1024, lr=5e-4):
    Xr, Xp, Ya = build_windows(buffer, H)
    xr = torch.tensor((Xr - norm["ref_mu"]) / norm["ref_sd"], device=dev)
    xp = torch.tensor((Xp - norm["pro_mu"]) / norm["pro_sd"], device=dev)
    ya = torch.tensor((Ya - norm["act_mu"]) / norm["act_sd"], device=dev)
    vae.to(dev).train()
    opt = torch.optim.AdamW(vae.parameters(), lr=lr)
    n = len(xr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            ah, mu, logvar = vae(xr[b], xp[b], sample=True)
            loss = ((ah - ya[b]) ** 2).mean() + beta * kl_divergence(mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
    vae.eval()
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm-start-ckpt", required=True, help="phase-2 sweep run dir (control_vae.pt)")
    p.add_argument("--model-folder", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--out", default="stage2/out/dagger")
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--clips-per-iter", type=int, default=40)
    p.add_argument("--epochs-per-iter", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    vae, H, norm = load_warm(args.warm_start_ckpt)
    vae.to(dev)
    print(f"DAgger: warm-start {args.warm_start_ckpt} | H={H} latent={vae.latent} | iters={args.iters}")

    @torch.no_grad()
    def student_action(Rfull, t, proprio):
        idx = np.minimum(np.arange(t, t + H), len(Rfull) - 1)
        w = ((Rfull[idx].reshape(-1) - norm["ref_mu"]) / norm["ref_sd"]).astype(np.float32)
        s = ((proprio - norm["pro_mu"]) / norm["pro_sd"]).astype(np.float32)
        mu, _ = vae.encode(torch.tensor(w, device=dev)[None])
        a = vae.decode(mu, torch.tensor(s, device=dev)[None]).cpu().numpy()[0]
        return (a * norm["act_sd"] + norm["act_mu"]).astype(np.float32)

    model = load_model_from_checkpoint_dir(Path(args.model_folder) / "checkpoint", device=dev)
    model.to(dev).eval()
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
    n_clips = min(args.clips_per_iter, len(joblib.load(args.data_path)))

    def aggregate_clip(mid):
        """Roll out the CURRENT student; log (s, r, a*=BFM.act(s,z)) per step. Returns clip dict + survived."""
        env.set_is_evaluating(mid)
        _, od = get_backward_observation(env, 0, use_root_height_obs=use_rh)
        obs_bwd, _ = get_backward_observation(env, 0, use_root_height_obs=use_rh)
        z_bfm = encode_z(tree_map(lambda x: x[1:], obs_bwd))
        ref_frames = tree_map(lambda x: x[1:], obs_bwd)
        T = z_bfm.shape[0]
        Rfull = np.stack([flatten_obs(tree_map(lambda x: x[t], ref_frames)) for t in range(T)])

        wrapped_env.reset(to_numpy=False)
        root_init = torch.cat([od["ref_body_pos"][0, 0], od["ref_body_rots"][0, 0],
                               od["ref_body_vels"][0, 0], od["ref_body_angular_vels"][0, 0]])
        dof_init = torch.zeros_like(env.simulator.dof_state.view(1, -1, 2)[0])
        dof_init[..., 0] = od["dof_pos"][0]; dof_init[..., 1] = od["ref_dof_vel"][0]
        wrapped_env._env.reset_envs_idx(env_ids, target_states={"dof_states": dof_init, "root_states": root_init[None]})
        wrapped_env.step(torch.zeros((1, act_dim), dtype=torch.float32), to_numpy=False)
        observation = wrapped_env._get_g1env_observation(to_numpy=False)

        S, R, A = [], [], []
        min_z = 1e9
        for i in range(min(T, args.max_steps)):
            s_vec = flatten_obs(observation)
            a_star = model.act(observation, z_bfm[i % T].repeat(1, 1), mean=True)   # TEACHER label at student state
            S.append(s_vec); R.append(Rfull[min(i, T - 1)])
            A.append(np.asarray(a_star.detach().cpu().numpy(), np.float32).reshape(-1)[:act_dim])
            # STUDENT drives the robot
            a_env = torch.tensor(student_action(Rfull, i, s_vec), dtype=torch.float32)[None]
            observation, *_ = wrapped_env.step(a_env, to_numpy=False)
            qz = float(np.asarray(wrapped_env._get_qpos_qvel(to_numpy=True)[0]).reshape(-1)[2])
            min_z = min(min_z, qz)
        clip = {"s": np.stack(S).astype(np.float32), "r": np.stack(R).astype(np.float32),
                "a": np.stack(A).astype(np.float32)}
        return clip, float(min_z > FALL_Z)

    buffer, curve = [], []
    for it in range(args.iters):
        surv = []
        for mid in range(n_clips):
            try:
                clip, s_ok = aggregate_clip(mid)
                buffer.append(clip); surv.append(s_ok)
            except Exception as e:
                print(f"  iter{it} clip{mid} FAIL {repr(e)[:100]}")
        student_surv = float(np.mean(surv)) if surv else 0.0
        n_samp = train_on_buffer(vae, buffer, norm, H, args.epochs_per_iter, dev, beta=args.beta)
        curve.append({"iter": it, "student_survival": round(student_surv, 3),
                      "buffer_clips": len(buffer), "train_samples": int(n_samp)})
        print(f"[iter {it}] student survival={student_surv:.3f} | buffer={len(buffer)} clips, {n_samp} samples")
        torch.save({"state_dict": vae.state_dict(),
                    "arch": {"ref_dim": vae.ref_dim, "proprio_dim": vae.proprio_dim,
                             "act_dim": vae.act_dim, "latent": vae.latent, "horizon": H}},
                   out / "control_vae_dagger.pt")
        np.savez(out / "normalization.npz", **norm)
        json.dump({"curve": curve, "warm_start": args.warm_start_ckpt},
                  open(out / "dagger_curve.json", "w"), indent=2)

    print("\n=== DAgger survival curve ===")
    for c in curve:
        print(f"  iter {c['iter']}: survival {c['student_survival']}  (buffer {c['buffer_clips']} clips)")


if __name__ == "__main__":
    main()
