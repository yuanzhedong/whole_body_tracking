"""ONE test of the 'detect-early + freeze-before-thrash' idea.
Run policy on DECODED (bad) motion with EARLY failure detection (tight threshold) and freeze
(stiffness=0 + stop driving) the instant an env trips — BEFORE it thrashes. Measure env.step time
in windows across the run. If it stays ~40ms (vs ~2000ms late/no-freeze), the idea works.

  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u stage2/bench_earlyfreeze.py [--no_freeze] [--late]
"""
import os, sys, time
sys.path.insert(0, "/ws/user/yzdong/src/github/whole_body_tracking/scripts/rsl_rl")
from isaaclab.app import AppLauncher
import cli_args
def log(*a): print(*a, flush=True)
WBT="/ws/user/yzdong/src/github/whole_body_tracking"
import argparse
p=argparse.ArgumentParser()
p.add_argument("--task",default="Tracking-Flat-G1-v0"); p.add_argument("--num_envs",type=int,default=128)
p.add_argument("--teacher_ckpt",default=f"{WBT}/logs/rsl_rl/g1_flat/2026-05-30_15-59-30_walk_4090_full30k/model_29999.pt")
p.add_argument("--motion",default=f"{WBT}/stage2/out/sim2sim_et_decoded/walk1_subject1_0_33_decoded.npz")
p.add_argument("--steps",type=int,default=200)
p.add_argument("--no_freeze",action="store_true")
p.add_argument("--keep_resets",action="store_true",help="keep fall-terminations ON (reset-on-fall) to measure reset overhead")
p.add_argument("--late",action="store_true",help="use the LATE (normal) threshold instead of early")
cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p)
args,hydra=p.parse_known_args(); args.headless=True; sys.argv=[sys.argv[0]]+hydra
app=AppLauncher(args).app
import torch, gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import whole_body_tracking.tasks
from rsl_rl.runners import OnPolicyRunner
@hydra_task_config(args.task,"rsl_rl_cfg_entry_point")
def main(env_cfg,agent_cfg):
    agent_cfg=cli_args.parse_rsl_rl_cfg(args.task,args)
    env_cfg.scene.num_envs=args.num_envs
    env_cfg.commands.motion.motion_file=args.motion
    if not args.keep_resets:
        for t in ["anchor_pos","anchor_ori","ee_body_pos","time_out"]: setattr(env_cfg.terminations,t,None)
        env_cfg.episode_length_s=1.0e6
    env=gym.make(args.task,cfg=env_cfg,render_mode=None); env=RslRlVecEnvWrapper(env); uenv=env.unwrapped
    runner=OnPolicyRunner(env,agent_cfg.to_dict(),log_dir=None,device=agent_cfg.device)
    runner.load(args.teacher_ckpt); policy=runner.get_inference_policy(device=uenv.device)
    from whole_body_tracking.tasks.tracking.mdp import terminations as T
    from isaaclab.managers import SceneEntityCfg
    rc=SceneEntityCfg("robot"); robot=uenv.scene["robot"]; dev=uenv.device; nj=robot.num_joints
    # EARLY threshold (catch the tip before the violent fall) vs LATE (normal)
    pos_th, ori_th = (0.25, 0.8) if args.late else (0.10, 0.4)
    mode = ("LATE" if args.late else "EARLY") + (" no-freeze" if args.no_freeze else " +freeze")
    log(f"\n=== EARLY-FREEZE TEST [{mode}] thresh pos>{pos_th} ori>{ori_th}, {args.num_envs} envs ===")
    ever=torch.zeros(args.num_envs,dtype=torch.bool,device=dev)
    obs,_=env.get_observations()
    win=20; t_acc=0.0; n_in=0
    for s in range(args.steps):
        torch.cuda.synchronize(); t0=time.time()
        with torch.inference_mode():
            a=policy(obs)
            if not args.no_freeze and ever.any():
                a=torch.where(ever.unsqueeze(-1), torch.zeros_like(a), a)
            obs,_,_,_=env.step(a)
        torch.cuda.synchronize(); t_acc+=(time.time()-t0)*1000; n_in+=1
        fail=(T.bad_anchor_pos_z_only(uenv,"motion",pos_th)|T.bad_anchor_ori(uenv,rc,"motion",ori_th))
        prev=ever.clone(); ever|=fail
        if not args.no_freeze:
            nids=torch.where(ever&~prev)[0]
            if nids.numel()>0:
                z=torch.zeros((nids.numel(),nj),device=dev)
                robot.write_joint_stiffness_to_sim(z,env_ids=nids)
                robot.write_joint_velocity_to_sim(z,env_ids=nids)
                robot.write_root_velocity_to_sim(torch.zeros((nids.numel(),6),device=dev),env_ids=nids)
        if (s+1)%win==0:
            log(f"  steps {s-win+2:3d}-{s+1:3d}: {t_acc/n_in:7.1f} ms/step   failed={int(ever.sum()):3d}/{args.num_envs}")
            t_acc=0.0; n_in=0
    log(f"  final survival = {float((~ever).float().mean()):.3f}")
    env.close()
if __name__=="__main__": main(); app.close()
