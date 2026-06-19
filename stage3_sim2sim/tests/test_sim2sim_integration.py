"""L2 integration: run the real HoloMotion tracker (MuJoCo) on a real clip.

Slow + heavy: needs the OMG repo, its venv, MuJoCo, and the HoloMotion ONNX. Skips
cleanly when any are missing. Validates the run_tracker wrapper + rollout_metrics
on the actual stack (Gate B: original motion survives).
"""
import glob
import os
from pathlib import Path
import numpy as np
import pytest

from stage3_sim2sim.sim2sim import build_qpos36_from_artifact, run_tracker, rollout_metrics

OMG_ROOT = "/ws/user/yzdong/src/github/OMG"
ONNX = ("/scratch/user/yzdong/OMG-models/holomotion_dl/"
        "HoloMotion_motion_tracking_model_v1.3.1/exported/motion_tracking_model.onnx")
ART_GLOB = "/scratch/user/yzdong/OMG-Data/raw/bones_seed/artifacts_seed_full/*walk*/motion.npz"


requires_stack = pytest.mark.skipif(
    not (Path(ONNX).exists() and Path(OMG_ROOT, ".venv-cu128/bin/python").exists()
         and glob.glob(ART_GLOB)),
    reason="HoloMotion ONNX / OMG venv / seed artifacts not present",
)


@requires_stack
@pytest.mark.slow
def test_gate_b_original_survives(tmp_path, capsys):
    """Gate B (validate-the-validator): the tracker executes ORIGINAL motion upright."""
    art = sorted(glob.glob(ART_GLOB))[0]
    qpos = build_qpos36_from_artifact(art)[:128]
    rollout = run_tracker(qpos, fps=30, out_dir=tmp_path / "gateB",
                          onnx_path=ONNX, omg_root=OMG_ROOT, num_frames=80)
    d = np.load(rollout)
    assert "executed_qpos_36" in d and "reference_qpos_36" in d
    m = rollout_metrics(d["executed_qpos_36"], d["reference_qpos_36"])
    with capsys.disabled():
        print(f"\n[Gate B] survival={m['survival']:.2f} root_z_min={m['root_z_min']:.3f} "
              f"joint_track={m['joint_rmse_deg']:.1f}deg")
    assert m["survival"] > 0.9, f"original motion did not survive: {m}"
    assert m["root_z_min"] > 0.4
