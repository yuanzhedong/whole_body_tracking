"""Sim-to-sim validation of the G1-VAE (Option A: shared-projection OmniMM modality).

Validates that the G1-VAE preserves enough motion information for a downstream
BeyondMimic tracking policy to still execute the motion well on the robot in
Isaac Sim.

Three phases:
  Phase 0 – Offline reconstruction (no Isaac):
    Encode clip features → latent z → decode → measure per-dim RMSE.
    Metrics: joint angle error (rad), root orient error (geodesic deg), root vel error (m/s).
    Target: joint angle RMSE < 0.1 rad (OmniMM biomech benchmark).

  Phase 1 – Kinematic FK replay (Isaac, no physics):
    Write decoded joint angles frame-by-frame into the kinematic sim to obtain
    body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w for ALL 30 bodies.
    Saves decoded motion as <out_dir>/<clip_name>_decoded.npz — a valid motion file
    compatible with MotionCommand / Tracking-Flat-G1-v0.

  Phase 2 – Tracking comparison (Isaac + RL policy):
    Run the Stage-0 tracking policy on ORIGINAL reference motion → baseline metrics.
    Run the same policy on the VAE-DECODED motion     → degraded metrics.
    Delta tells you exactly how much the VAE compression costs in tracking quality.
    Verdict: VAE is "good enough" if decoded survival ≥ 0.90 × original and
             decoded E_mpbpe ≤ 1.15 × original.

Coordinate systems:
  G1 dataset is exported in y-up (OmniMM convention).
  Isaac Sim runs in z-up.
  Conversion: T_YUP_TO_ZUP = T_ZUP_TO_YUP.T = [[1,0,0],[0,0,-1],[0,1,0]].
  Joint angles are frame-relative DoF — no conversion needed.
  Root position and orientation need the inverse basis change.

Usage (4090, headless):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=YES \\
    .venv/bin/python stage2/sim2sim_vae_eval.py \\
      --vae_ckpt UniMoTok/experiments/biomechanics_tokenizer/G1_MldVAE_v1/checkpoints/epoch=564.ckpt \\
      --dataset_dir stage2/out/g1_dataset_yup \\
      --teacher_ckpt logs/rsl_rl/g1_flat/2026-05-30_15-59-30_walk_4090_full30k/model_29999.pt \\
      --splits val test \\
      --out stage2/out/sim2sim_eval.json
"""
import argparse, os, sys, json, types
import numpy as np

# ── parse args before any Isaac imports so --phase0_only skips AppLauncher ──
parser = argparse.ArgumentParser()
parser.add_argument("--task",        default="Tracking-Flat-G1-v0")
parser.add_argument("--num_envs",    type=int,   default=128)
parser.add_argument("--vae_ckpt",    required=True)
parser.add_argument("--dataset_dir", default="stage2/out/g1_dataset_yup")
parser.add_argument("--splits",      nargs="+", default=["val", "test"])
parser.add_argument("--teacher_ckpt",required=True)
parser.add_argument("--out",         default="stage2/out/sim2sim_eval.json")
parser.add_argument("--decoded_dir", default=None)
parser.add_argument("--eval_reps",   type=int,   default=2)
parser.add_argument("--phase0_only", action="store_true",
                    help="Phase 0 offline reconstruction only — no Isaac Sim needed")
args, remaining = parser.parse_known_args()

if not args.phase0_only:
    sys.path.append("/ws/user/yzdong/src/github/whole_body_tracking/scripts/rsl_rl")
    from isaaclab.app import AppLauncher
    import cli_args
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    args.headless = True
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app

sys.path.append("/ws/user/yzdong/src/github/whole_body_tracking/UniMoTok")

import torch
from scipy.spatial.transform import Rotation

# ── coordinate system constants ────────────────────────────────────────────
# G1 dataset is y-up; Isaac is z-up.
T_YUP_TO_ZUP = np.array([[1, 0,  0],
                          [0, 0,  1],
                          [0, -1, 0]], dtype=np.float64)   # inverse of T_ZUP_TO_YUP

FEATURE_LAYOUT = {"root_rot6d": (0, 6), "root_lin_vel": (6, 9),
                  "root_ang_vel": (9, 12), "joint_pos": (12, 41)}
FEATURE_DIM = 41
N_JOINTS    = 29


# ── helpers ─────────────────────────────────────────────────────────────────

def load_vae(ckpt_path, device="cpu"):
    from multimodal_tokenizers.archs.mld_vae import MldVaeBiomechanics
    c   = torch.load(ckpt_path, map_location=device, weights_only=False)
    ta  = c["hyper_parameters"]["tokenizer_arch"]
    args_ns = types.SimpleNamespace(**ta["params"])
    vae = MldVaeBiomechanics(args_ns).to(device)
    # strip "vae." prefix from Lightning state dict
    sd = {k[4:]: v for k, v in c["state_dict"].items() if k.startswith("vae.")}
    vae.load_state_dict(sd)
    vae.eval()
    return vae


def load_norm(dataset_dir):
    p = np.load(os.path.join(dataset_dir, "normalization.npz"))
    return p["mean"].astype(np.float32), np.maximum(p["std"].astype(np.float32), 1e-6)


def load_clips(dataset_dir, splits):
    clips = []
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            print(f"  [warn] split dir not found: {split_dir}")
            continue
        for fname in sorted(os.listdir(split_dir)):
            if fname.endswith(".npz"):
                clips.append({"name": fname[:-4], "split": split,
                               "path": os.path.join(split_dir, fname)})
    return clips


def rot6d_to_quat_xyzw(rot6d):
    """rot6d [T,6] -> quaternion [T,4] xyzw (in source frame, heading NOT restored)."""
    r6 = rot6d.reshape(-1, 6)
    a1 = r6[:, :3]; a2 = r6[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-9)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    mat = np.stack([b1, b2, b3], axis=-1)         # [T,3,3] column-major
    return Rotation.from_matrix(mat).as_quat()    # xyzw


def integrate_root_position(root_lin_vel, dt, start_pos=None):
    """Integrate root velocity (heading frame) -> approximate root position."""
    T = root_lin_vel.shape[0]
    pos = np.zeros((T, 3), dtype=np.float32)
    if start_pos is not None:
        pos[0] = start_pos
    for t in range(1, T):
        pos[t] = pos[t-1] + root_lin_vel[t-1] * dt
    return pos


def yup_to_zup_quat(quat_xyzw):
    """Convert quaternion from y-up canonical frame to z-up world frame."""
    mats = Rotation.from_quat(quat_xyzw).as_matrix()
    mats_zup = T_YUP_TO_ZUP @ mats @ T_YUP_TO_ZUP.T
    return Rotation.from_matrix(mats_zup).as_quat()


def denormalize(feat, mean, std):
    return feat * std + mean


# ── Phase 0: offline reconstruction metrics ─────────────────────────────────

def phase0_reconstruction(vae, clips, mean, std, device):
    """Encode each clip → decode → measure reconstruction quality."""
    results = {}
    for clip in clips:
        d = np.load(clip["path"], allow_pickle=True)
        feat_orig = np.asarray(d["motion"], dtype=np.float32)         # [T, 41]
        T = feat_orig.shape[0]

        feat_norm = (feat_orig - mean) / std
        x = torch.from_numpy(feat_norm).unsqueeze(0).to(device)       # [1, T, 41]

        with torch.no_grad():
            out = vae(x)
            rec_norm = out["rec_pose"].squeeze(0).cpu().numpy()        # [T, 41]

        feat_rec = denormalize(rec_norm, mean, std)

        # per-component errors
        ja_rmse = float(np.sqrt(((feat_rec[:, 12:41] - feat_orig[:, 12:41])**2).mean()))
        rv_mae  = float(np.abs(feat_rec[:, 6:9]  - feat_orig[:, 6:9]).mean())
        av_mae  = float(np.abs(feat_rec[:, 9:12] - feat_orig[:, 9:12]).mean())

        # root orientation geodesic error (degrees)
        q_orig = rot6d_to_quat_xyzw(feat_orig[:, :6])
        q_rec  = rot6d_to_quat_xyzw(feat_rec[:, :6])
        R_orig = Rotation.from_quat(q_orig)
        R_rec  = Rotation.from_quat(q_rec)
        angle_err = (R_orig.inv() * R_rec).magnitude()                # [T] radians
        orient_err_deg = float(np.degrees(angle_err.mean()))

        results[clip["name"]] = {
            "split": clip["split"],
            "n_frames": int(T),
            "joint_angle_rmse_rad": round(ja_rmse, 5),
            "root_lin_vel_mae":     round(rv_mae,  5),
            "root_ang_vel_mae":     round(av_mae,  5),
            "root_orient_err_deg":  round(orient_err_deg, 3),
            "pass_joint_angle":     bool(ja_rmse < 0.10),             # OmniMM target
        }
        status = "PASS" if ja_rmse < 0.10 else "WARN"
        print(f"  [{status}] {clip['name']:40s}  jt_rmse={ja_rmse:.4f} rad  "
              f"orient={orient_err_deg:.2f}°  rv={rv_mae:.4f}")

        # cache decoded features for Phase 1
        clip["feat_orig"] = feat_orig
        clip["feat_rec"]  = feat_rec

    n_pass = sum(1 for r in results.values() if r["pass_joint_angle"])
    print(f"\nPhase 0: {n_pass}/{len(results)} clips meet < 0.10 rad joint angle target")
    return results


# ── Phase 1: kinematic FK replay → decoded motion npz ───────────────────────

def phase1_kinematic_replay(clips, decoded_dir, fps=50):
    """Use Isaac kinematic replay to reconstruct body_pos_w from decoded joint angles."""
    import gymnasium as gym
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import whole_body_tracking.tasks  # noqa: F401

    os.makedirs(decoded_dir, exist_ok=True)
    # Use a single env for kinematic stepping (no RL, physics disabled).
    # We set one env, drive joint states manually, read back body poses.
    from isaaclab.app import AppLauncher  # already launched

    # Import sim utilities
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    import whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg as g1_cfg

    # Build a minimal scene with just the G1 robot for FK
    from source.whole_body_tracking.whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingFlatEnvCfg
    env_cfg = g1_cfg.G1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.commands.motion.motion_file = clips[0]["path"].replace(
        "g1_dataset_yup", "artifacts").replace("_yup/val/", ":v0/").replace(
        "_yup/test/", ":v0/").replace(".npz", "/motion.npz")   # fallback

    # Actually: use a real motion file to init the env, then overwrite joint states
    # We'll find any valid motion file
    import glob
    fallback_motion = glob.glob("artifacts/*/motion.npz")[0]

    env_cfg.commands.motion.motion_file = fallback_motion
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env_wrapped = RslRlVecEnvWrapper(env)
    uenv = env_wrapped.unwrapped
    robot = uenv.scene["robot"]
    dev   = uenv.device

    # joint order in the sim (must match feature vector order)
    from stage2.export_g1_motion import JOINT_NAMES as FEAT_JOINT_NAMES
    sim_jnames = list(robot.data.joint_names)
    feat_to_sim = [sim_jnames.index(n) for n in FEAT_JOINT_NAMES]

    TARGET_FPS = fps

    for clip in clips:
        if "feat_rec" not in clip:
            continue  # phase 0 must have run first

        feat_rec  = clip["feat_rec"]                  # [T,41] y-up decoded
        feat_orig = clip["feat_orig"]                 # [T,41] y-up original
        T = feat_rec.shape[0]
        dt = 1.0 / TARGET_FPS

        # Extract components
        joint_angles = feat_rec[:, 12:41]             # [T,29]
        root_lin_vel = feat_rec[:, 6:9]               # [T,3] y-up heading frame
        root_rot6d   = feat_rec[:, :6]                # [T,6] y-up

        # Reconstruct root trajectory (y-up) then convert to z-up
        root_pos_yup = integrate_root_position(root_lin_vel, dt,
                                               start_pos=np.array([0.0, 0.75, 0.0]))
        root_pos_zup = (T_YUP_TO_ZUP @ root_pos_yup.T).T

        q_xyzw_yup   = rot6d_to_quat_xyzw(root_rot6d)   # [T,4] heading-canonical y-up
        q_wxyz_zup   = Rotation.from_quat(
            yup_to_zup_quat(q_xyzw_yup)).as_quat()[[..., [3, 0, 1, 2]]]  # wxyz z-up
        # fix indexing:
        R_yup = Rotation.from_quat(q_xyzw_yup)
        R_zup = Rotation.from_matrix(T_YUP_TO_ZUP @ R_yup.as_matrix() @ T_YUP_TO_ZUP.T)
        q_wxyz_zup = R_zup.as_quat()[:, [3, 0, 1, 2]]   # xyzw->wxyz

        # Reset env, drive kinematically
        env_wrapped.reset()
        log = {k: [] for k in ["joint_pos", "joint_vel", "body_pos_w",
                                "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]}

        for t in range(T):
            # Write root state
            root_state = torch.zeros(1, 13, device=dev)
            root_state[0, :3]  = torch.tensor(root_pos_zup[t], device=dev)
            root_state[0, 3:7] = torch.tensor(q_wxyz_zup[t],   device=dev)
            # velocities zero (kinematic)
            robot.write_root_state_to_sim(root_state)

            # Write joint positions
            jpos = robot.data.default_joint_pos.clone()
            jpos[0, feat_to_sim] = torch.tensor(joint_angles[t], dtype=torch.float32, device=dev)
            robot.write_joint_state_to_sim(jpos, robot.data.default_joint_vel.clone())
            uenv.sim.step(render=False)
            robot.update(dt)

            log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy())
            log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy())
            log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy())
            log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy())

        out_path = os.path.join(decoded_dir, clip["name"] + "_decoded.npz")
        np.savez(out_path,
                 fps=np.array([TARGET_FPS]),
                 joint_pos=np.stack(log["joint_pos"]),
                 joint_vel=np.stack(log["joint_vel"]),
                 body_pos_w=np.stack(log["body_pos_w"]),
                 body_quat_w=np.stack(log["body_quat_w"]),
                 body_lin_vel_w=np.stack(log["body_lin_vel_w"]),
                 body_ang_vel_w=np.stack(log["body_ang_vel_w"]))
        clip["decoded_npz"] = out_path
        print(f"  saved {clip['name']}_decoded.npz  [{T} frames @ {TARGET_FPS} fps]")

    env_wrapped.close()


# ── Phase 2: tracking comparison ────────────────────────────────────────────

MET = ["error_body_pos", "error_joint_pos", "error_anchor_pos"]

def run_tracking(env_cfg, agent_cfg, policy, motion_path, n_reps, device):
    """Run policy on motion_path for n_reps * clip_length steps; return metrics."""
    import gymnasium as gym
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env_cfg_copy = env_cfg.__class__()
    env_cfg_copy.__dict__.update(env_cfg.__dict__)
    env_cfg_copy.commands.motion.motion_file = motion_path

    env = gym.make(args.task, cfg=env_cfg_copy, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    uenv = env.unwrapped
    cmd  = uenv.command_manager.get_term("motion")
    tm   = uenv.termination_manager
    mlen = int(cmd.motion.time_step_total)

    n_fail = n_comp = 0
    sums = {k: 0.0 for k in MET}
    csum = 0
    obs, _ = env.get_observations()
    for _ in range(mlen * n_reps):
        with torch.inference_mode():
            a = policy(obs)
            obs, _, _, _ = env.step(a)
        n_fail += int(tm.terminated.sum())
        n_comp += int(tm.time_outs.sum())
        for k in MET:
            sums[k] += cmd.metrics[k].mean().item()
        csum += 1

    env.close()
    tot = n_fail + n_comp
    return {
        "survival":      round((n_comp / tot) if tot else float("nan"), 4),
        "e_mpbpe_mm":    round(sums["error_body_pos"]  / csum * 1000, 2),
        "e_mpjpe_rad":   round(sums["error_joint_pos"] / csum, 4),
        "e_anchor_mm":   round(sums["error_anchor_pos"]/ csum * 1000, 2),
    }


def phase2_tracking_comparison(clips, teacher_ckpt, eval_reps):
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from rsl_rl.runners import OnPolicyRunner
    import whole_body_tracking.tasks  # noqa: F401
    import whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg as g1_cfg

    # Load policy once
    env_cfg  = g1_cfg.G1FlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    # dummy motion file to init
    import glob
    env_cfg.commands.motion.motion_file = glob.glob("artifacts/*/motion.npz")[0]
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env_wrapped = RslRlVecEnvWrapper(env)
    from rsl_rl.runners import OnPolicyRunner
    agent_cfg_obj = cli_args.parse_rsl_rl_cfg(args.task, args)
    runner = OnPolicyRunner(env_wrapped, agent_cfg_obj.to_dict(),
                            log_dir=None, device=env_wrapped.unwrapped.device)
    runner.load(teacher_ckpt)
    policy = runner.get_inference_policy(device=env_wrapped.unwrapped.device)
    env_wrapped.close()

    results = {}
    for clip in clips:
        if "decoded_npz" not in clip:
            print(f"  skip {clip['name']}: no decoded npz (Phase 1 must run first)")
            continue

        # Find original motion file in artifacts
        orig_path = os.path.join("artifacts", clip["name"] + ":v0", "motion.npz")
        if not os.path.exists(orig_path):
            print(f"  [warn] original motion not found: {orig_path}")
            continue

        print(f"  evaluating {clip['name']}...")
        m_orig    = run_tracking(env_cfg, agent_cfg_obj, policy, orig_path,    eval_reps, None)
        m_decoded = run_tracking(env_cfg, agent_cfg_obj, policy, clip["decoded_npz"], eval_reps, None)

        surv_ratio = (m_decoded["survival"]   / m_orig["survival"]   if m_orig["survival"]   > 0 else 0)
        mpbpe_ratio= (m_decoded["e_mpbpe_mm"] / m_orig["e_mpbpe_mm"] if m_orig["e_mpbpe_mm"] > 0 else 999)

        verdict = (surv_ratio >= 0.90 and mpbpe_ratio <= 1.15)
        results[clip["name"]] = {
            "split": clip["split"],
            "original":      m_orig,
            "decoded":       m_decoded,
            "survival_ratio":round(surv_ratio,  3),
            "mpbpe_ratio":   round(mpbpe_ratio, 3),
            "pass":          verdict,
        }
        status = "PASS" if verdict else "FAIL"
        print(f"  [{status}] {clip['name']}")
        print(f"    survival: orig={m_orig['survival']:.3f}  decoded={m_decoded['survival']:.3f}  ratio={surv_ratio:.3f}")
        print(f"    mpbpe:    orig={m_orig['e_mpbpe_mm']:.1f}mm  decoded={m_decoded['e_mpbpe_mm']:.1f}mm  ratio={mpbpe_ratio:.3f}")

    n_pass = sum(1 for r in results.values() if r["pass"])
    print(f"\nPhase 2: {n_pass}/{len(results)} clips PASS tracking comparison")
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    decoded_dir = args.decoded_dir or (args.out.replace(".json", "_decoded"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading G1-VAE from {args.vae_ckpt}")
    vae  = load_vae(args.vae_ckpt, device=device)
    mean, std = load_norm(args.dataset_dir)
    clips = load_clips(args.dataset_dir, args.splits)
    print(f"Loaded {len(clips)} clips from splits {args.splits}")

    output = {"vae_ckpt": args.vae_ckpt, "dataset_dir": args.dataset_dir,
              "splits": args.splits, "clips": {}}

    # ── Phase 0 ─────────────────────────────────────────────────────────────
    print("\n── Phase 0: Offline reconstruction metrics ─────────────────────")
    p0 = phase0_reconstruction(vae, clips, mean, std, device)
    for name, r in p0.items():
        output["clips"].setdefault(name, {}).update({"phase0": r})

    if args.phase0_only:
        json.dump(output, open(args.out, "w"), indent=2)
        print(f"\nPhase 0 results → {args.out}")
        return

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    print("\n── Phase 1: Kinematic FK replay → decoded motion npz ──────────")
    phase1_kinematic_replay(clips, decoded_dir, fps=50)
    for clip in clips:
        if "decoded_npz" in clip:
            output["clips"].setdefault(clip["name"], {})["decoded_npz"] = clip["decoded_npz"]

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    print("\n── Phase 2: Tracking comparison (orig vs decoded) ──────────────")
    p2 = phase2_tracking_comparison(clips, args.teacher_ckpt, args.eval_reps)
    for name, r in p2.items():
        output["clips"].setdefault(name, {}).update({"phase2": r})

    # ── summary verdict ──────────────────────────────────────────────────────
    p0_pass = sum(1 for c in output["clips"].values()
                  if c.get("phase0", {}).get("pass_joint_angle"))
    p2_pass = sum(1 for c in output["clips"].values()
                  if c.get("phase2", {}).get("pass"))
    n = len(output["clips"])
    output["summary"] = {
        "n_clips": n,
        "phase0_joint_angle_pass": f"{p0_pass}/{n}",
        "phase2_tracking_pass":    f"{p2_pass}/{n}",
        "overall_verdict": "PASS" if (p0_pass == n and p2_pass >= n * 0.8) else "FAIL",
    }
    print(f"\n{'='*60}")
    print(f"SUMMARY: Phase0 {p0_pass}/{n} pass | Phase2 {p2_pass}/{n} pass")
    print(f"OVERALL: {output['summary']['overall_verdict']}")

    json.dump(output, open(args.out, "w"), indent=2)
    print(f"Results → {args.out}")


main()
if not args.phase0_only:
    app.close()  # type: ignore[name-defined]
