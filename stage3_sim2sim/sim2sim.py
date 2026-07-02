"""L2/L3 sim2sim driver: qpos_36 -> HoloMotion tracker (MuJoCo) -> survival/tracking metrics.

The tracker + physics already exist in the OMG repo; this module builds the
ground-truth qpos, shells out to the OMG ``tracker-only`` pipeline, and scores the
rollout. ``rollout_metrics`` is pure-numpy and unit-tested; ``run_tracker`` is an
integration wrapper that needs the OMG env + HoloMotion ONNX (skipped if absent).
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
import numpy as np

FALL_HEIGHT = 0.4  # root z below this = fallen (standing G1 pelvis ~0.75 m)
REL_MARGIN = 0.15  # executed pelvis may sit this far below the REFERENCE pelvis before counting as fallen


def build_qpos36_from_artifact(artifact_npz):
    """Ground-truth z-up qpos_36 from an artifacts/<name>/motion.npz.

    qpos_36 = [root_pos(3), root_quat_wxyz(4), joint_pos(29)].
    """
    d = np.load(artifact_npz, allow_pickle=True)
    root = d["body_pos_w"][:, 0, :]      # z-up pelvis
    quat = d["body_quat_w"][:, 0, :]     # wxyz
    joints = d["joint_pos"][:, :29]
    return np.concatenate([root, quat, joints], axis=1).astype(np.float32)


def build_hybrid_qpos36(decoded_features, original_qpos36, joint_slice=slice(12, 41)):
    """Reference for sim2sim = VAE-**decoded joints** + **original root** pose.

    The VAE reconstructs joints + root tilt faithfully, but the integrated world
    root (height/trajectory) is corrupted by an upstream double-yup convention in
    the dataset feature build (see decode_to_qpos36 / SIM2SIM_PLAN.md). Pairing
    decoded joints with the clip's true root isolates the question that matters —
    *is the decoded joint motion physically executable?* — without contaminating it
    with a root-reconstruction artifact.
    """
    decoded_features = np.asarray(decoded_features, dtype=np.float32)
    original_qpos36 = np.asarray(original_qpos36, dtype=np.float32)
    n = min(len(decoded_features), len(original_qpos36))
    q = original_qpos36[:n].copy()
    q[:, 7:36] = decoded_features[:n, joint_slice]   # replace 29 joints with decoded
    return q


def rollout_metrics(executed_qpos36, reference_qpos36, fall_height=FALL_HEIGHT,
                    rel_margin=REL_MARGIN):
    """Score a tracker rollout (executed vs reference qpos_36).

    Two survival notions are returned:

    * ``survival`` — absolute: fraction of frames the executed pelvis stays above a
      fixed ``fall_height`` (0.4 m). Simple, but **too crude for legitimately-low
      motions**: a correct deep squat/crouch reference pelvis dips to ~0.27-0.44 m,
      so this metric flags it as "fallen" even when tracking is perfect.
    * ``survival_rel`` — reference-relative: fraction of frames the executed pelvis
      stays within ``rel_margin`` (0.15 m) **below the reference pelvis**, i.e.
      ``executed_z > reference_z - rel_margin``. This credits holding the intended
      (possibly low) posture and only penalizes an actual collapse, so it is the
      fair survival metric for near-ground motion. For standing motions the two
      agree; they diverge exactly where the absolute metric is misleading.

    Also returns min/mean root height, joint tracking RMSE (deg), pelvis-height MAE
    vs reference, and root horizontal drift (m).
    """
    ex = np.asarray(executed_qpos36, dtype=np.float64)
    ref = np.asarray(reference_qpos36, dtype=np.float64)
    n = min(len(ex), len(ref))
    ex, ref = ex[:n], ref[:n]
    root_z, ref_z = ex[:, 2], ref[:, 2]
    survival = float((root_z > fall_height).mean())
    survival_rel = float((root_z > ref_z - rel_margin).mean())
    joint_rmse_deg = float(np.sqrt(np.mean((ex[:, 7:36] - ref[:, 7:36]) ** 2)) * 180 / np.pi)
    root_xy = np.linalg.norm(ex[:, :2] - ref[:, :2], axis=1)
    return {
        "survival": survival,
        "survival_rel": survival_rel,
        "root_z_min": float(root_z.min()),
        "root_z_mean": float(root_z.mean()),
        "ref_z_min": float(ref_z.min()),
        "root_z_mae": float(np.abs(root_z - ref_z).mean()),
        "joint_rmse_deg": joint_rmse_deg,
        "root_xy_drift_mean": float(root_xy.mean()),
        "root_xy_drift_max": float(root_xy.max()),
        "n_frames": int(n),
    }


def run_tracker(qpos36, fps, out_dir, onnx_path, omg_root, num_frames=150,
                providers="CPUExecutionProvider", timeout=900):
    """Run the OMG ``tracker-only`` pipeline on a qpos_36 sequence; return rollout npz path.

    Shells out to the OMG repo's pipeline (separate venv + PYTHONPATH=src + MuJoCo).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qpos_npy = out_dir / "seed_qpos36.npy"
    np.save(qpos_npy, np.asarray(qpos36, dtype=np.float32))

    omg_root = Path(omg_root)
    py = omg_root / ".venv-cu128" / "bin" / "python"
    env = dict(os.environ)
    env.update(PYTHONPATH="src", MUJOCO_GL="egl", TOKENIZERS_PARALLELISM="false",
               CUDA_VISIBLE_DEVICES="")
    env_file = omg_root / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                env.setdefault(k.strip(), v.strip())
    cmd = [str(py), "-m", "omg.cli.pipeline.main", "--mode", "tracker-only",
           "--seed-motion", str(qpos_npy), "--seed-fps", str(fps),
           "--holomotion-onnx", str(onnx_path), "--tracker-providers", providers,
           "--num-frames", str(num_frames), "--output-root", str(out_dir)]
    subprocess.run(cmd, cwd=str(omg_root), env=env, check=True,
                   timeout=timeout, capture_output=True)
    hits = list(out_dir.rglob("holomotion_rollout.npz"))
    if not hits:
        raise RuntimeError(f"tracker produced no rollout under {out_dir}")
    return hits[0]
