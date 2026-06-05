"""Phase-1 of BeyondMimic Stage-2: distill a frozen tracking policy (teacher) into a
conditional motion VAE via DAgger (Fig 7B-i, Table S6).

The policy observation (PolicyCfg) splits into:
  reference-motion terms  -> ENCODER input  (command/phase + anchor pose error)
  proprioceptive terms    -> DECODER input  (base/joint vel, joint pos, last action)
We roll out a DAgger mixture (teacher -> student over training) in Tracking-Flat-G1-v0,
query the teacher for the supervised action, and train the VAE to reconstruct it + KL.

Runs on the Isaac 4.5 stack, headless, on a 4090 (no Blackwell kernels):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
    .venv/bin/python stage2/distill_vae.py --teacher_ckpt <walk model_29999.pt> \
        --motion_file /tmp/wbt_fix/walk.npz --iters 10000 --out stage2/out/vae_walk.pt
"""
import argparse, os, sys, json, time
sys.path.append("/ws/user/yzdong/src/github/whole_body_tracking/scripts/rsl_rl")
sys.path.append("/ws/user/yzdong/src/github/whole_body_tracking/stage2")
from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--teacher_ckpt", required=True)
parser.add_argument("--motion_file", required=True)
parser.add_argument("--iters", type=int, default=10000)
parser.add_argument("--latent", type=int, default=32)
parser.add_argument("--beta", type=float, default=0.01)        # Table S6 KL coef
parser.add_argument("--lr", type=float, default=5e-4)          # Table S6
parser.add_argument("--accum", type=int, default=15)           # Table S6 accumulated grad steps
parser.add_argument("--dagger_anneal", type=int, default=3000) # steps to go teacher->student
parser.add_argument("--out", default="stage2/out/vae_walk.pt")
parser.add_argument("--log", default="stage2/out/distill.log")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True                       # box's RTX renderer crashes on window creation
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import whole_body_tracking.tasks  # noqa: F401,E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from vae_model import MotionVAE, kl_divergence  # noqa: E402

PROPRIO_START_TERM = "base_lin_vel"  # first proprioceptive term; everything before it is reference


def _logline(path, msg):
    with open(path, "a") as f:
        f.write(msg + "\n"); f.flush(); os.fsync(f.fileno())


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
    dev = env.unwrapped.device

    # --- split the policy observation into reference vs proprioceptive parts ---
    om = env.unwrapped.observation_manager
    names = om.active_terms["policy"] if hasattr(om, "active_terms") else om.group_obs_term_names["policy"]
    dims = om.group_obs_term_dim["policy"]
    flat = [int(torch.tensor(d).prod()) for d in dims]
    split = names.index(PROPRIO_START_TERM)
    ref_dim = sum(flat[:split])
    proprio_dim = sum(flat[split:])
    act_dim = env.num_actions
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.log, "w").close()
    _logline(args.log, f"obs terms={names}")
    _logline(args.log, f"REF terms={names[:split]} ({ref_dim}d) | PROPRIO terms={names[split:]} ({proprio_dim}d) | act={act_dim}")

    vae = MotionVAE(ref_dim, proprio_dim, act_dim, latent=args.latent).to(dev)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)

    obs, _ = env.get_observations()
    t0 = time.time(); run_recon = run_kl = 0.0
    opt.zero_grad()
    for it in range(args.iters):
        ref, proprio = obs[:, :ref_dim], obs[:, ref_dim:]
        with torch.inference_mode():
            teacher_a = teacher(obs).clone()
        student_a, mu, logvar = vae(ref, proprio)
        recon = ((student_a - teacher_a) ** 2).sum(-1).mean()      # ||a_hat - a||^2
        kl = kl_divergence(mu, logvar)
        (recon + args.beta * kl).div(args.accum).backward()
        if (it + 1) % args.accum == 0:
            opt.step(); opt.zero_grad()
        run_recon += recon.item(); run_kl += kl.item()

        # DAgger: step a teacher->student mixture so the env visits the student's distribution
        mix = max(0.0, 1.0 - it / args.dagger_anneal)
        with torch.no_grad():
            step_a = mix * teacher_a + (1.0 - mix) * student_a.detach()
        obs, _, _, _ = env.step(step_a)

        if (it + 1) % 100 == 0:
            n = 100
            _logline(args.log, f"it {it+1:6d}/{args.iters} | recon {run_recon/n:.4f} | kl {run_kl/n:.3f} "
                               f"| mix {mix:.2f} | {(it+1)/(time.time()-t0):.1f} it/s")
            run_recon = run_kl = 0.0
        if (it + 1) % 2000 == 0:
            torch.save({"state_dict": vae.state_dict(), "ref_dim": ref_dim, "proprio_dim": proprio_dim,
                        "act_dim": act_dim, "latent": args.latent}, args.out)

    torch.save({"state_dict": vae.state_dict(), "ref_dim": ref_dim, "proprio_dim": proprio_dim,
                "act_dim": act_dim, "latent": args.latent}, args.out)
    _logline(args.log, f"DONE -> {args.out}")
    print("DISTILL_DONE", args.out)
    env.close()


main()
app.close()
