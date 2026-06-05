#!/usr/bin/env bash
# Render a trained tracking policy to an MP4 on this machine.
#
# Pipeline (decoupled, because Isaac Sim 4.5's RTX renderer crashes on driver 595):
#   1) rollout the policy headless on the 4.5 training stack (.venv)   -> per-frame states .npz
#   2) replay+render those states in Isaac Sim 6.0 (.venv6)            -> PNG frames
#   3) ffmpeg                                                          -> MP4
#
# Usage:
#   ./render_policy.sh --ckpt logs/rsl_rl/g1_flat/<run>/model_299.pt \
#       --motion cs224n-robustqa/wandb-registry-motions/walk1_subject1 \
#       [--out video.mp4] [--steps 500] [--camera follow|treadmill] [--fps 25] [--gpu 0]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

CKPT=""; MOTION=""; OUT="policy_render.mp4"; STEPS=500; CAMERA="treadmill"; FPS=25; GPU=0; TASK="Tracking-Flat-G1-v0"; WANDB_RUN=""; CAPTION=""; PHASE0=""
while [[ $# -gt 0 ]]; do case "$1" in
  --phase0) PHASE0="--start_phase0"; shift 1;;   # start rollout at motion frame 0 (clean from-the-start clip)
  --ckpt) CKPT="$2"; shift 2;;
  --motion) MOTION="$2"; shift 2;;
  --out) OUT="$2"; shift 2;;
  --steps) STEPS="$2"; shift 2;;
  --camera) CAMERA="$2"; shift 2;;
  --fps) FPS="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;
  --task) TASK="$2"; shift 2;;
  --wandb_run) WANDB_RUN="$2"; shift 2;;   # entity/project/run_id -> upload video to that run
  --caption) CAPTION="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 1;;
esac; done
[[ -z "$CKPT" || -z "$MOTION" ]] && { echo "ERROR: --ckpt and --motion are required" >&2; exit 1; }

set -a; [[ -f .env ]] && source .env; set +a
export OMNI_KIT_ACCEPT_EULA=YES UV_LINK_MODE=copy CUDA_VISIBLE_DEVICES="$GPU"
WORK="$(mktemp -d /tmp/render_policy.XXXXXX)"
STATES="$WORK/states.npz"; FRAMES="$WORK/frames"; RES="$WORK/render.txt"

echo "[1/3] resolving motion + rollout (Isaac Sim 4.5 stack, headless)..."
# resolve motion: local .npz used as-is; otherwise treat as a wandb registry artifact name
if [[ "$MOTION" == *.npz && -f "$MOTION" ]]; then
  MOTION_NPZ="$MOTION"
else
  MOTION_NPZ="$WORK/motion.npz"
  .venv/bin/python - "$MOTION" "$MOTION_NPZ" <<'PY'
import sys, shutil, pathlib, wandb
name, dst = sys.argv[1], sys.argv[2]
if ":" not in name: name += ":latest"
art = wandb.Api().artifact(name)
shutil.copy(pathlib.Path(art.download()) / "motion.npz", dst)
print("motion ->", dst)
PY
fi

.venv/bin/python tools/rollout_log.py --task="$TASK" --num_envs 1 --steps "$STEPS" \
  --ckpt "$CKPT" --motion_file "$MOTION_NPZ" --out "$STATES" --headless $PHASE0
echo "    states: $STATES"

echo "[2/3] rendering in Isaac Sim 6.0 (.venv6)..."
.venv6/bin/python tools/render_rollout_sim6.py --states "$STATES" --out_dir "$FRAMES" \
  --usd_dir "$WORK/g1_usd" --camera "$CAMERA" --result "$RES" --res 1280 720
grep -q RENDER_OK "$RES" || { echo "ERROR: render failed:"; cat "$RES"; exit 1; }

echo "[3/3] encoding MP4 -> $OUT ..."
ffmpeg -y -framerate "$FPS" -pattern_type glob -i "$FRAMES/rgb_*.png" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT" 2>/dev/null
echo "DONE: $OUT"
ls -la "$OUT"

if [[ -n "$WANDB_RUN" ]]; then
  echo "[+] uploading video to WandB run $WANDB_RUN ..."
  .venv/bin/python tools/upload_video_wandb.py --run "$WANDB_RUN" --video "$OUT" \
    --key "media/policy_render" --caption "${CAPTION:-$(basename "$OUT")}" --fps "$FPS"
fi
