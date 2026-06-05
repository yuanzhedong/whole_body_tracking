"""Verify a distilled motion VAE BEFORE moving to the diffusion stage (Phase 1 gate).

Four gates, each compared against the teacher tracking policy:
  G1 Reconstruction  : ||decode(mu=E(ref), proprio) - a_teacher|| on teacher-distribution states.
  G2 Closed-loop     : run the VAE AS a policy (a = decode(E(ref), proprio)) and measure the
                       tracking metrics (success rate, E_mpbpe, E_mpjpe) vs the teacher. THE gate.
  G3 Latent structure: aggregate-posterior stats, per-dim KL / active dims, and a z-ablation
                       (||decode(z=E(ref)) - decode(z=0)||) to confirm the decoder USES the latent
                       (else the diffusion has nothing to model). Diffusion-readiness.
  G4 Robustness      : closed-loop with added action noise -> recovery / fall rate (Phase-2 pre-check).

Usage (4090, headless):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
    .venv/bin/python stage2/verify_vae.py --vae stage2/out/vae_walk.pt \
        --teacher_ckpt <walk model_29999.pt> --motion_file /tmp/wbt_fix/walk.npz \
        --out stage2/out/verify_walk.json
"""
import argparse, os, sys, json
sys.path.append("/ws/user/yzdong/src/github/whole_body_tracking/scripts/rsl_rl")
sys.path.append("/ws/user/yzdong/src/github/whole_body_tracking/stage2")
from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--vae", required=True)
parser.add_argument("--teacher_ckpt", required=True)
parser.add_argument("--motion_file", required=True)
parser.add_argument("--recon_steps", type=int, default=400)   # G1/G3 collection (teacher-driven)
parser.add_argument("--eval_episodes", type=float, default=4.0)  # G2/G4 closed-loop length (motion-lengths)
parser.add_argument("--noise_std", type=float, default=0.1)   # G4 OU-like action noise scale
parser.add_argument("--out", default="stage2/out/verify_vae.json")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import whole_body_tracking.tasks  # noqa: F401,E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from vae_model import MotionVAE  # noqa: E402

PROPRIO_START_TERM = "base_lin_vel"
MET = ["error_body_pos", "error_joint_pos", "error_anchor_pos"]


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.motion.motion_file = args.motion_file
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.teacher_ckpt)
    teacher = runner.get_inference_policy(device=env.unwrapped.device)
    uenv = env.unwrapped; dev = uenv.device
    cmd = uenv.command_manager.get_term("motion")
    tm = uenv.termination_manager

    om = uenv.observation_manager
    names = om.active_terms["policy"] if hasattr(om, "active_terms") else om.group_obs_term_names["policy"]
    dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim["policy"]]
    split = names.index(PROPRIO_START_TERM); ref_dim = sum(dims[:split])

    ck = torch.load(args.vae, map_location=dev)
    vae = MotionVAE(ck["ref_dim"], ck["proprio_dim"], ck["act_dim"], latent=ck["latent"]).to(dev)
    vae.load_state_dict(ck["state_dict"]); vae.eval()

    def vae_action(obs):
        mu, _ = vae.encode(obs[:, :ref_dim])
        return vae.decode(mu, obs[:, ref_dim:])          # deterministic (mean z)

    def closed_loop(policy, motion_lens, noise=0.0):
        """run `policy` as the controller; return success rate + tracking errors."""
        mlen = int(cmd.motion.time_step_total); steps = int(mlen * motion_lens)
        nfail = ncomp = csum = 0; sums = {k: 0.0 for k in MET}
        obs, _ = env.get_observations()
        for _ in range(steps):
            with torch.inference_mode():
                a = policy(obs)
                if noise: a = a + noise * torch.randn_like(a)
                obs, _, _, _ = env.step(a)
            nfail += int(tm.terminated.sum()); ncomp += int(tm.time_outs.sum())
            for k in MET: sums[k] += cmd.metrics[k].mean().item()
            csum += 1
        tot = nfail + ncomp
        return {"success_rate": (ncomp / tot) if tot else float("nan"),
                "E_mpbpe_mm": sums["error_body_pos"] / csum * 1000,
                "E_mpjpe_rad": sums["error_joint_pos"] / csum,
                "E_anchor_mm": sums["error_anchor_pos"] / csum * 1000}

    out = {"vae": args.vae}

    # ---- G1 reconstruction + G3 latent stats: roll out the TEACHER, probe the VAE ----
    mus, recon_mean, recon_samp, abl = [], 0.0, 0.0, 0.0; n = 0
    klsum = None
    obs, _ = env.get_observations()
    for _ in range(args.recon_steps):
        with torch.inference_mode():
            ta = teacher(obs)
            ref, pro = obs[:, :ref_dim], obs[:, ref_dim:]
            mu, logvar = vae.encode(ref)
            a_mean = vae.decode(mu, pro)
            a_samp = vae.decode(vae.reparameterize(mu, logvar), pro)
            a_zero = vae.decode(torch.zeros_like(mu), pro)
            recon_mean += (a_mean - ta).pow(2).sum(-1).mean().item()
            recon_samp += (a_samp - ta).pow(2).sum(-1).mean().item()
            abl += ((a_mean - a_zero).norm(dim=-1) / (ta.norm(dim=-1) + 1e-6)).mean().item()
            kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)   # per-dim
            klsum = kl if klsum is None else klsum + kl
            mus.append(mu); obs, _, _, _ = env.step(ta)   # step teacher (on-distribution)
        n += 1
    mus = torch.cat(mus, 0); kl = klsum / n
    out["G1_reconstruction"] = {"recon_mse_mean_z": recon_mean / n, "recon_mse_sampled_z": recon_samp / n}
    out["G3_latent"] = {
        "agg_posterior_mu_mean_abs": mus.mean(0).abs().mean().item(),
        "agg_posterior_mu_std_mean": mus.std(0).mean().item(),
        "active_dims_kl>0.01": int((kl > 0.01).sum()), "latent_dim": ck["latent"],
        "per_dim_kl_max": float(kl.max()), "per_dim_kl_mean": float(kl.mean()),
        "z_ablation_rel_change": abl / n,   # ||decode(E(ref)) - decode(0)|| / ||a||  -> z must matter
    }

    # ---- G2 closed-loop tracking: VAE policy vs teacher ----
    out["G2_closed_loop_vae"] = closed_loop(vae_action, args.eval_episodes)
    out["G2_closed_loop_teacher"] = closed_loop(teacher, args.eval_episodes)
    # ---- G4 robustness: VAE policy + action noise ----
    out["G4_robustness_vae_noisy"] = closed_loop(vae_action, args.eval_episodes, noise=args.noise_std)

    # ---- verdicts (relative to teacher) ----
    t, v = out["G2_closed_loop_teacher"], out["G2_closed_loop_vae"]
    out["verdict"] = {
        "G2_tracks": v["success_rate"] >= 0.9 * t["success_rate"] and v["E_mpbpe_mm"] <= 1.3 * t["E_mpbpe_mm"],
        "G3_latent_used": out["G3_latent"]["z_ablation_rel_change"] >= 0.1,
        "G3_not_collapsed": out["G3_latent"]["active_dims_kl>0.01"] >= 2,
        "G1_reconstructs": out["G1_reconstruction"]["recon_mse_mean_z"] < 1.0,  # tune to action scale
    }
    out["verdict"]["PASS_to_diffusion"] = all(out["verdict"][k] for k in
        ["G2_tracks", "G3_latent_used", "G3_not_collapsed"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    sys.stderr.write("VERIFY_RESULTS " + json.dumps(out) + "\n"); sys.stderr.flush()
    print("VERIFY_DONE", json.dumps(out))
    env.close()


main()
app.close()
