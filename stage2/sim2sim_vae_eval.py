"""Sim-to-sim validation of the G1-VAE (Option A: shared-projection OmniMM modality).

Three phases:
  Phase 0 – Offline reconstruction (pure numpy/VAE):
    Encode clip → latent z → decode → per-dim RMSE.
    Target: joint RMSE < 0.10 rad (OmniMM biomech benchmark).

  Phase 1 – Build decoded motion npz:
    Splice decoded joint angles into original motion npz (root kept from original).
    No FK replay needed — root trajectory differences are small for reconstruction
    quality testing; tracking policy is primarily sensitive to joint angle reference.

  Phase 2 – Tracking comparison (Isaac + RL policy), NO-RESET survival by default:
    Run Stage-0 tracking on ORIGINAL reference → baseline.
    Run Stage-0 tracking on VAE-DECODED reference → degraded metrics.
    Fall-terminations are disabled so a fallen robot is NOT reset (no free recovery);
    survival = fraction of envs that never fell over one clip pass. (--reset_based for old.)
    Verdict: decoded survival ≥ 0.90 × original AND mpbpe ≤ 1.15 × original.

CRITICAL STRUCTURE NOTE: All three phases run INSIDE the @hydra_task_config context,
which must wrap the outermost entry point — exactly as in eval_tracking_quality.py and
verify_vae.py. Calling gym.make outside that context causes a carb re-init crash.

Usage:
  # Phase 0+1 only (no Isaac):
  .venv/bin/python stage2/sim2sim_vae_eval.py --phase01_only \\
      --vae_ckpt ... --dataset_dir ... --teacher_ckpt dummy

  # Full pipeline (all three phases, requires Isaac):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=YES \\
    .venv/bin/python stage2/sim2sim_vae_eval.py \\
      --vae_ckpt ... --dataset_dir ... --teacher_ckpt ... \\
      --splits val test --out stage2/out/sim2sim_eval.json
"""
import argparse, os, sys, json, types
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/ws/user/yzdong/src/github/whole_body_tracking/scripts/rsl_rl")
sys.path.insert(0, "/ws/user/yzdong/src/github/whole_body_tracking/UniMoTok")

parser = argparse.ArgumentParser()
parser.add_argument("--task",         default="Tracking-Flat-G1-v0")
parser.add_argument("--num_envs",     type=int, default=128)
parser.add_argument("--vae_ckpt",     required=True)
parser.add_argument("--dataset_dir",  default="stage2/out/g1_dataset_yup")
parser.add_argument("--artifacts_dir",default="artifacts")
parser.add_argument("--splits",       nargs="+", default=["val", "test"])
parser.add_argument("--teacher_ckpt", required=True)
parser.add_argument("--out",          default="stage2/out/sim2sim_eval.json")
parser.add_argument("--decoded_dir",  default=None)
parser.add_argument("--eval_reps",    type=int, default=2)
parser.add_argument("--max_steps",    type=int, default=500,
                    help="hard cap on Phase-2 sim steps per run. Default 500 (=10s @50Hz) — a robot "
                         "that will fall does so in <~2s, and one surviving 10s survives the clip, so "
                         "500 steps is a strong survival proxy AND bounds runtime. 0 = full clip.")
parser.add_argument("--fail_mult",    type=float, default=8.0,
                    help="early-exit a hopeless run once cumulative env failures exceed "
                         "num_envs*fail_mult (a well-tracked run never approaches this; a "
                         "constantly-falling decoded run trips it in a few hundred steps)")
parser.add_argument("--min_steps",    type=int, default=200,
                    help="warmup steps before the fail-budget early-exit can fire")
parser.add_argument("--no_reset", action="store_true", default=True,
                    help="NO-RESET single-pass eval (DEFAULT): disable the fall-triggering "
                         "termination terms so a fallen robot is NOT reset, run one clip-length "
                         "pass, detect falls manually with the same thresholds. survival = fraction "
                         "of envs that never fell. This is the HONEST survival metric — reset-based "
                         "eval teleports a fallen robot back onto the reference, letting it 'recover' "
                         "and inflating survival (e.g. same bad clip: reset-based 0.74 vs no-reset "
                         "0.36). Speed is NOT the reason: measured ~28-32ms/step either way on an "
                         "idle box, even at 0%% survival — the old '~50x reset-on-fall' / 'thrashing "
                         "~2000ms/step' slowdowns were CPU contention from co-located training, not "
                         "PhysX. See OVERNIGHT_RESULTS.md. Use --reset_based for the old metric.")
parser.add_argument("--reset_based", dest="no_reset", action="store_false",
                    help="reset-based Phase-2: fallen robots are teleported back onto the reference "
                         "and continue; survival = time-outs / (time-outs + terminations). Inflates "
                         "survival via free recovery. Kept for comparison only.")
parser.add_argument("--phase01_only", action="store_true",
                    help="Phase 0+1 only — skips Isaac entirely")
# Pre-parse to check phase01_only before any Isaac imports
_pre, _ = parser.parse_known_args()

if not _pre.phase01_only:
    try:
        import cli_args as _cli; _cli.add_rsl_rl_args(parser)
    except Exception: pass
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    args.headless = True
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app
    import whole_body_tracking.tasks  # registers Tracking-Flat-G1-v0 with gymnasium
else:
    args, _ = parser.parse_known_args()
    app = None

import torch

# ── constants ─────────────────────────────────────────────────────────────────
FEAT_DIM  = 41
N_JOINTS  = 29
MET       = ["error_body_pos", "error_joint_pos", "error_anchor_pos"]
FEAT_JOINT_NAMES = [
    "left_hip_pitch_joint","right_hip_pitch_joint","waist_yaw_joint",
    "left_hip_roll_joint","right_hip_roll_joint","waist_roll_joint",
    "left_hip_yaw_joint","right_hip_yaw_joint","waist_pitch_joint",
    "left_knee_joint","right_knee_joint","left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint","left_ankle_pitch_joint","right_ankle_pitch_joint",
    "left_shoulder_roll_joint","right_shoulder_roll_joint","left_ankle_roll_joint",
    "right_ankle_roll_joint","left_shoulder_yaw_joint","right_shoulder_yaw_joint",
    "left_elbow_joint","right_elbow_joint","left_wrist_roll_joint",
    "right_wrist_roll_joint","left_wrist_pitch_joint","right_wrist_pitch_joint",
    "left_wrist_yaw_joint","right_wrist_yaw_joint",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_vae(ckpt_path):
    from multimodal_tokenizers.archs.mld_vae import MldVaeBiomechanics
    c   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ta  = c["hyper_parameters"]["tokenizer_arch"]
    vae = MldVaeBiomechanics(types.SimpleNamespace(**ta["params"]))
    sd  = {k[4:]: v for k,v in c["state_dict"].items() if k.startswith("vae.")}
    vae.load_state_dict(sd); vae.eval()
    return vae


def load_norm(dataset_dir):
    p = np.load(os.path.join(dataset_dir, "normalization.npz"))
    return p["mean"].astype(np.float32), np.maximum(p["std"].astype(np.float32), 1e-6)


def load_clips(dataset_dir, splits):
    clips = []
    for split in splits:
        d = os.path.join(dataset_dir, split)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".npz"):
                clips.append({"name": f[:-4], "split": split,
                               "path": os.path.join(d, f)})
    return clips


def rot6d_to_quat(rot6d):
    r = rot6d.reshape(-1, 6)
    b1 = r[:,:3] / (np.linalg.norm(r[:,:3], axis=-1, keepdims=True) + 1e-9)
    b2 = r[:,3:] - (b1*r[:,3:]).sum(-1, keepdims=True)*b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-9)
    return Rotation.from_matrix(np.stack([b1, b2, np.cross(b1,b2)], -1)).as_quat()


def resample(x, n):
    T = x.shape[0]
    if T == n: return x
    t_in = np.linspace(0,1,T); t_out = np.linspace(0,1,n)
    flat = x.reshape(T,-1)
    return np.stack([np.interp(t_out,t_in,flat[:,c]) for c in range(flat.shape[1])],1).reshape(n,*x.shape[1:])


# ── Phase 0 ───────────────────────────────────────────────────────────────────

def phase0(vae, clips, mean, std, window_size=128, stride=64):
    results = {}
    for clip in clips:
        d    = np.load(clip["path"], allow_pickle=True)
        feat = np.asarray(d["motion"], dtype=np.float32)  # [T, 41]
        T    = feat.shape[0]
        feat_n = (feat - mean) / std  # normalised

        # Slide 128-frame windows (same as training dataloader), decode each, stitch by average
        rec_n = np.zeros_like(feat_n)
        count = np.zeros(T, dtype=np.float32)
        starts = list(range(0, max(1, T - window_size + 1), stride))
        if not starts or starts[-1] + window_size > T:
            starts = [max(0, T - window_size)]  # one tail window for short clips
        for s in starts:
            end = s + window_size
            win = feat_n[s:end]
            if len(win) < window_size:
                win = np.pad(win, ((0, window_size - len(win)), (0, 0)))
            x = torch.from_numpy(win).unsqueeze(0)  # [1, 128, 41]
            with torch.no_grad():
                r = vae(x)["rec_pose"].squeeze(0).numpy()  # [128, 41]
            actual = min(window_size, T - s)
            rec_n[s:s+actual] += r[:actual]
            count[s:s+actual] += 1
        count = np.maximum(count, 1)
        rec = (rec_n / count[:, None]) * std + mean  # unnormalise

        ja  = float(np.sqrt(((rec[:,12:]-feat[:,12:])**2).mean()))
        rv  = float(np.abs(rec[:,6:9]-feat[:,6:9]).mean())
        od  = float(np.degrees((Rotation.from_quat(rot6d_to_quat(feat[:,:6])).inv() *
                                 Rotation.from_quat(rot6d_to_quat(rec[:,:6]))).magnitude().mean()))
        clip["feat_rec"] = rec
        results[clip["name"]] = {"split":clip["split"],"n_frames":feat.shape[0],
            "joint_angle_rmse_rad":round(ja,5),"root_lin_vel_mae":round(rv,5),
            "root_orient_err_deg":round(od,3),"pass_joint_angle":bool(ja<0.10)}
        print(f"  [{'PASS' if ja<0.10 else 'WARN'}] {clip['name']:42s}  "
              f"jt_rmse={ja:.4f} rad  orient={od:.2f}°")
    print(f"\nPhase 0: {sum(r['pass_joint_angle'] for r in results.values())}/{len(results)} "
          f"pass < 0.10 rad")
    return results


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def phase1(clips, decoded_dir, artifacts_dir):
    os.makedirs(decoded_dir, exist_ok=True)
    for clip in clips:
        if "feat_rec" not in clip: continue
        orig = os.path.join(artifacts_dir, clip["name"]+":v0", "motion.npz")
        if not os.path.exists(orig):
            print(f"  [skip] {clip['name']}: not in {artifacts_dir}"); continue
        d = np.load(orig, allow_pickle=True)
        T = d["joint_pos"].shape[0]
        joints_50fps = resample(clip["feat_rec"][:,12:], T).astype(np.float32)
        jp = d["joint_pos"].copy(); jp[:,:] = joints_50fps
        out = os.path.join(decoded_dir, clip["name"]+"_decoded.npz")
        np.savez(out, fps=d["fps"], joint_pos=jp, joint_vel=d["joint_vel"],
                 body_pos_w=d["body_pos_w"], body_quat_w=d["body_quat_w"],
                 body_lin_vel_w=d["body_lin_vel_w"], body_ang_vel_w=d["body_ang_vel_w"])
        clip["decoded_npz"] = out
        print(f"  saved {clip['name']}_decoded.npz  [{T} frames]")


# ── Phase 2 inside hydra context ─────────────────────────────────────────────

def phase2_inside_hydra(clips, env_cfg, agent_cfg, artifacts_dir, eval_reps):
    import gymnasium as gym
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    env_cfg.scene.num_envs = args.num_envs
    device = agent_cfg.device
    results = {}

    for clip in clips:
        if "decoded_npz" not in clip:
            print(f"  skip {clip['name']}: no decoded npz"); continue
        orig = os.path.join(artifacts_dir, clip["name"]+":v0", "motion.npz")
        if not os.path.exists(orig):
            print(f"  skip {clip['name']}: original not found"); continue

        def run_one(path):
            env_cfg.commands.motion.motion_file = path
            env    = gym.make(args.task, cfg=env_cfg, render_mode=None)
            env    = RslRlVecEnvWrapper(env)
            uenv   = env.unwrapped
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
            runner.load(args.teacher_ckpt)
            policy = runner.get_inference_policy(device=device)
            cmd, tm = uenv.command_manager.get_term("motion"), uenv.termination_manager
            mlen = int(cmd.motion.time_step_total)
            n_steps = mlen * eval_reps
            if args.max_steps > 0:
                n_steps = min(n_steps, args.max_steps)
            fail_budget = args.num_envs * args.fail_mult  # hopeless-run early-exit threshold
            nf=nc=ns=0; sums={k:0. for k in MET}
            bailed = False
            obs, _ = env.get_observations()
            for _ in range(n_steps):
                with torch.inference_mode(): a=policy(obs); obs,_,_,_=env.step(a)
                nf+=int(tm.terminated.sum()); nc+=int(tm.time_outs.sum())
                for k in MET: sums[k]+=cmd.metrics[k].mean().item()
                ns+=1
                # Early exit: a constantly-falling (untrackable) run racks up failures fast.
                # A well-tracked run keeps nf≈0 and never trips this.
                if ns >= args.min_steps and nf > fail_budget:
                    bailed = True
                    break
            env.close()
            tot = nf+nc
            return {"survival":round((nc/tot) if tot else float("nan"),4),
                    "e_mpbpe_mm":round(sums["error_body_pos"]/ns*1000,2),
                    "e_mpjpe_rad":round(sums["error_joint_pos"]/ns,4),
                    "e_anchor_mm":round(sums["error_anchor_pos"]/ns*1000,2),
                    "steps_run":ns,"steps_planned":n_steps,"early_exit":bailed,
                    "n_terminated":nf,"n_completed":nc}

        def run_one_noreset(path):
            # Disable fall-triggering terminations so a fallen robot is NOT reset back onto the
            # reference. Run ONE clip pass; detect falls manually with the SAME threshold functions.
            # survival = fraction of envs that NEVER fell — the honest metric (reset-based eval lets
            # a fallen robot teleport back and "recover", inflating survival). A fallen robot left
            # un-reset just ragdolls; on an idle box this stays ~28-32ms/step (the prior "reset/
            # thrashing ~2000ms" was CPU contention from co-located training, not PhysX).
            env_cfg.commands.motion.motion_file = path
            env_cfg.terminations.anchor_pos = None
            env_cfg.terminations.anchor_ori = None
            env_cfg.terminations.ee_body_pos = None
            env_cfg.terminations.time_out = None
            env_cfg.episode_length_s = 1.0e6  # no time-based truncation within one pass
            env    = gym.make(args.task, cfg=env_cfg, render_mode=None)
            env    = RslRlVecEnvWrapper(env)
            uenv   = env.unwrapped
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
            runner.load(args.teacher_ckpt)
            policy = runner.get_inference_policy(device=device)
            cmd    = uenv.command_manager.get_term("motion")
            mlen   = int(cmd.motion.time_step_total)
            n_steps = mlen if args.max_steps <= 0 else min(mlen, args.max_steps)
            from whole_body_tracking.tasks.tracking.mdp import terminations as T
            from isaaclab.managers import SceneEntityCfg
            robot_cfg = SceneEntityCfg("robot")
            ne = uenv.num_envs
            ever_failed = torch.zeros(ne, dtype=torch.bool, device=device)
            sums = {k: 0. for k in MET}; nm = 0
            bailed = False
            obs, _ = env.get_observations()
            for step_idx in range(n_steps):
                with torch.inference_mode():
                    a = policy(obs)
                    obs, _, _, _ = env.step(a)
                # Detect falls with the SAME thresholds the termination terms would use.
                fail = (T.bad_anchor_pos_z_only(uenv, "motion", 0.25) |
                        T.bad_anchor_ori(uenv, robot_cfg, "motion", 0.8))
                ever_failed |= fail  # sticky: once an env falls it stays failed for this pass
                # metrics over ALIVE envs only (failed envs have meaningless huge error)
                alive = ~ever_failed
                if bool(alive.any()):
                    for k in MET: sums[k] += float(cmd.metrics[k][alive].mean())
                    nm += 1
                # Survival-decided early-exit: once >=95% have EVER failed, survival<=5% is locked
                # in (ever_failed is sticky) — stop. Good runs keep ever_failed≈0 -> full clip pass.
                # This bounds step COUNT only; per-step cost is already ~28-32ms regardless of falls.
                if (step_idx + 1) >= args.min_steps and float(ever_failed.float().mean()) >= 0.95:
                    bailed = True
                    break
            survival = float((~ever_failed).float().mean())
            ns = max(nm, 1)
            steps_run = step_idx + 1
            env.close()
            return {"survival": round(survival, 4),
                    "e_mpbpe_mm": round(sums["error_body_pos"]/ns*1000, 2),
                    "e_mpjpe_rad": round(sums["error_joint_pos"]/ns, 4),
                    "e_anchor_mm": round(sums["error_anchor_pos"]/ns*1000, 2),
                    "steps_run": steps_run, "steps_planned": n_steps, "early_exit": bailed,
                    "n_failed": int(ever_failed.sum()), "n_envs": ne}

        _run = run_one_noreset if args.no_reset else run_one
        mode = "no-reset single-pass" if args.no_reset else "reset-based"
        print(f"  evaluating {clip['name']}  ({mode})...")
        mo = _run(orig)
        print(f"    original: survival={mo['survival']} steps={mo['steps_run']}/{mo['steps_planned']}")
        md = _run(clip["decoded_npz"])
        print(f"    decoded:  survival={md['survival']} steps={md['steps_run']}/{md['steps_planned']}")
        sr = md["survival"]/mo["survival"] if mo["survival"]>0 else 0
        mr = md["e_mpbpe_mm"]/mo["e_mpbpe_mm"] if mo["e_mpbpe_mm"]>0 else 999
        verdict = sr>=0.90 and mr<=1.15
        results[clip["name"]] = {"split":clip["split"],"original":mo,"decoded":md,
            "survival_ratio":round(sr,3),"mpbpe_ratio":round(mr,3),"pass":verdict}
        print(f"  [{'PASS' if verdict else 'FAIL'}] {clip['name']}: "
              f"survival ×{sr:.2f}  mpbpe ×{mr:.2f}")

    n = sum(1 for r in results.values() if r["pass"])
    print(f"\nPhase 2: {n}/{len(results)} clips PASS")
    return results


# ── entry point ───────────────────────────────────────────────────────────────

if args.phase01_only:
    def main():
        _run(None, None)

    def _run(env_cfg, agent_cfg):
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        decoded_dir = args.decoded_dir or args.out.replace(".json", "_decoded")
        vae        = load_vae(args.vae_ckpt)
        mean, std  = load_norm(args.dataset_dir)
        clips      = load_clips(args.dataset_dir, args.splits)
        print(f"Loaded {len(clips)} clips from {args.splits}")
        output = {"vae_ckpt":args.vae_ckpt,"splits":args.splits,"clips":{}}
        print("\n── Phase 0 ──────────────────────────────────────────────────")
        for n,r in phase0(vae, clips, mean, std).items():
            output["clips"].setdefault(n,{})["phase0"]=r
        print("\n── Phase 1 ──────────────────────────────────────────────────")
        phase1(clips, decoded_dir, args.artifacts_dir)
        for c in clips:
            if "decoded_npz" in c:
                output["clips"].setdefault(c["name"],{})["decoded_npz"]=c["decoded_npz"]
        json.dump(output, open(args.out,"w"), indent=2)
        print(f"\nResults → {args.out}")
    main()

else:
    from isaaclab_tasks.utils.hydra import hydra_task_config

    @hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        """All phases run inside this hydra context — required for gym.make to work."""
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        decoded_dir = args.decoded_dir or args.out.replace(".json", "_decoded")
        vae        = load_vae(args.vae_ckpt)
        mean, std  = load_norm(args.dataset_dir)
        clips      = load_clips(args.dataset_dir, args.splits)
        print(f"Loaded {len(clips)} clips from {args.splits}")
        output = {"vae_ckpt":args.vae_ckpt,"splits":args.splits,"clips":{}}

        print("\n── Phase 0: Offline reconstruction ─────────────────────────")
        for n,r in phase0(vae, clips, mean, std).items():
            output["clips"].setdefault(n,{})["phase0"]=r

        print("\n── Phase 1: Build decoded motion npz ───────────────────────")
        phase1(clips, decoded_dir, args.artifacts_dir)
        for c in clips:
            if "decoded_npz" in c:
                output["clips"].setdefault(c["name"],{})["decoded_npz"]=c["decoded_npz"]

        print("\n── Phase 2: Tracking comparison ────────────────────────────")
        p2 = phase2_inside_hydra(clips, env_cfg, agent_cfg, args.artifacts_dir, args.eval_reps)
        for n,r in p2.items():
            output["clips"].setdefault(n,{})["phase2"]=r

        p0p = sum(1 for c in output["clips"].values() if c.get("phase0",{}).get("pass_joint_angle"))
        p2p = sum(1 for c in output["clips"].values() if c.get("phase2",{}).get("pass"))
        n   = len(output["clips"])
        output["summary"] = {"n_clips":n,
            "phase0_joint_rmse_pass":f"{p0p}/{n}",
            "phase2_tracking_pass":  f"{p2p}/{n}",
            "overall_verdict":"PASS" if p0p==n and p2p>=max(1,int(n*0.8)) else "FAIL"}
        print(f"\n{'='*55}")
        print(f"Phase 0: {p0p}/{n} | Phase 2: {p2p}/{n} | OVERALL: {output['summary']['overall_verdict']}")
        json.dump(output, open(args.out,"w"), indent=2)
        print(f"Results → {args.out}")

    main()
    app.close()
