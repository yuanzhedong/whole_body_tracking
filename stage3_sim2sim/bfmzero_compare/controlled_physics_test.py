"""Controlled test: run BFM-Zero with HoloMotion's EXACT per-joint PD gains.

BFM-Zero's deploy controller uses uniform kp=50/kd=5; HoloMotion uses per-joint gains
(knee 99, hip_pitch 40, ankle 28, ...). This forces BFM-Zero onto HoloMotion's gains
and checks whether it STILL holds the near-ground postures -- isolating the policy
from the physics/controller config. Run with CUDA_VISIBLE_DEVICES set.
"""
import os
os.environ["MUJOCO_GL"] = "egl"; os.environ["OMP_NUM_THREADS"] = "1"
import json, joblib, sys
from pathlib import Path
import numpy as np, torch
from torch.utils._pytree import tree_map
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.utils.helpers import get_backward_observation

# HoloMotion per-joint kp/kd in FEATURE order (from ONNX metadata)
ONNX = "/scratch/user/yzdong/OMG-models/holomotion_dl/HoloMotion_motion_tracking_model_v1.3.1/exported/motion_tracking_model.onnx"
import onnxruntime as ort
_m = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
_arr = lambda k: np.array([float(x) for x in _m[k].replace("[", " ").replace("]", " ").replace(",", " ").split()])
JN = _m["joint_names"].split(",")                       # FEATURE order
HOLO_KP_FEAT, HOLO_KD_FEAT = _arr("joint_stiffness"), _arr("joint_damping")
# BFM env p_gains are in MuJoCo XML order (= OMG order). reorder FEATURE -> OMG by name.
OMG_ORDER = ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint","waist_yaw_joint","waist_roll_joint","waist_pitch_joint","left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]
perm = [JN.index(n) for n in OMG_ORDER]
HOLO_KP, HOLO_KD = HOLO_KP_FEAT[perm], HOLO_KD_FEAT[perm]


def _set1(obj, kp, kd):
    if obj is None or not hasattr(obj, "p_gains"):
        return False
    pg = obj.p_gains
    if hasattr(pg, "detach"):   # torch tensor
        obj.p_gains = torch.tensor(kp, dtype=pg.dtype, device=pg.device)
        obj.d_gains = torch.tensor(kd, dtype=obj.d_gains.dtype, device=obj.d_gains.device)
    else:                        # numpy
        obj.p_gains = np.asarray(kp, dtype=pg.dtype)
        obj.d_gains = np.asarray(kd, dtype=obj.d_gains.dtype)
    return True


def set_all_gains(env, kp, kd):
    holders = [env, getattr(env, "simulator", None)]
    for o in (env, getattr(env, "simulator", None)):
        if o is not None:
            holders += [v for v in vars(o).values() if hasattr(v, "p_gains")]
    n = sum(_set1(o, kp, kd) for o in holders if o is not None)
    return n


def main(mode="holo_gains"):   # "holo_gains" or "native"
    mf = Path("pretrained/new_model_for_training_code_inference")
    model = load_model_from_checkpoint_dir(mf / "checkpoint", device="cuda"); model.to("cuda").eval()
    cfg = json.load(open(mf / "config.json")); use_rh = cfg["env"].get("root_height_obs", False)
    cfg["env"]["lafan_tail_path"] = str(Path("pretrained/data/quant_clips.pkl").resolve())
    cfg["env"]["hydra_overrides"] += ["env.config.max_episode_length_s=10000", "env.config.headless=True", "simulator=mujoco"]
    cfg["env"]["disable_domain_randomization"] = False; cfg["env"]["disable_obs_noise"] = False
    we, _ = HumanoidVerseIsaacConfig(**cfg["env"]).build(1); env = we._env; eids = torch.arange(1)
    pg0 = env.p_gains.detach().cpu().numpy()
    print(f"native env.p_gains: hip_pitch {pg0[0]:.0f} knee {pg0[3]:.0f} waist {pg0[12]:.0f}")
    if mode == "holo_gains":
        n = set_all_gains(env, HOLO_KP, HOLO_KD)
        print(f"-> patched {n} gain holders to HoloMotion: hip_pitch {HOLO_KP[0]:.0f} knee {HOLO_KP[3]:.0f} waist {HOLO_KP[12]:.0f}")

    for mid, name in [(0, "crouch(ref0.20)"), (1, "squat(ref0.27)"), (3, "squat2(ref0.22)")]:
        env.set_is_evaluating(mid)
        obs, od = get_backward_observation(env, 0, use_root_height_obs=use_rh)
        expert = np.concatenate([od["ref_body_pos"][:, 0].cpu().numpy(), np.roll(od["ref_body_rots"][:, 0].cpu().numpy(), 1, -1), od["dof_pos"].cpu().numpy()], -1)
        z = model.project_z(model.backward_map(tree_map(lambda x: x[1:], obs)))
        we.reset(to_numpy=False)
        if mode == "holo_gains":   # reset may restore gains
            set_all_gains(env, HOLO_KP, HOLO_KD)
        ri = torch.cat([od["ref_body_pos"][0, 0], od["ref_body_rots"][0, 0], od["ref_body_vels"][0, 0], od["ref_body_angular_vels"][0, 0]])
        di = torch.zeros_like(env.simulator.dof_state.view(1, -1, 2)[0]); di[..., 0] = od["dof_pos"][0]; di[..., 1] = od["ref_dof_vel"][0]
        we._env.reset_envs_idx(eids, target_states={"dof_states": di, "root_states": ri[None]})
        we.step(torch.zeros((1, we.action_space.shape[-1])), to_numpy=False)
        o = we._get_g1env_observation(to_numpy=False)
        ex = [np.asarray(we._get_qpos_qvel(to_numpy=True)[0]).reshape(-1)[:36]]
        for i in range(min(z.shape[0], 600)):
            a = model.act(o, z[i % len(z)].repeat(1, 1), mean=True); o, *_ = we.step(a, to_numpy=False)
            ex.append(np.asarray(we._get_qpos_qvel(to_numpy=True)[0]).reshape(-1)[:36])
        ex = np.stack(ex); ref = expert[:len(ex)]
        surv = (ex[:, 2] > ref[:len(ex), 2] - 0.15).mean()
        lk = 7 + OMG_ORDER.index("left_knee_joint")
        print(f"[{mode}] {name}: survival_rel {surv:.2f} | exec pelvis mean {ex[:,2].mean():.2f} | "
              f"L-knee max {np.degrees(np.abs(ex[:,lk])).max():.0f}deg")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "holo_gains")
