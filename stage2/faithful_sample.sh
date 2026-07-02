#!/bin/bash
# Faithful-pipeline convergence sample: train per-clip tracking policies on NEW diverse LAFAN1 clips
# to refine the per-clip iters-to-gate estimate. Each clip already converted+uploaded to registry.
# Capped at 15k iters (captures the gate for all tiers incl. hard ~10-12k). Instrumented stdout logs
# carry per-iter "Mean episode length" -> convergence curve. One clip per free 4090 (GPU 4,5,2).
set -u; cd /ws/user/yzdong/src/github/whole_body_tracking
REG="cs224n-robustqa/wandb-registry-Motions"
ITERS=${ITERS:-15000}
run() { # clip gpu runname log
  local clip=$1 gpu=$2 name=$3 log=$4
  echo "[$(date +%H:%M:%S)] START $clip GPU$gpu -> $name"
  WANDB_ENTITY=cs224n-robustqa CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu \
  OMNI_KIT_ACCEPT_EULA=YES OMNI_USER_HOME=/tmp/omni_$name \
  .venv/bin/python scripts/rsl_rl/train.py --task Tracking-Flat-G1-v0 --num_envs 2048 \
    --registry_name "$REG/lafan_$clip:latest" --max_iterations $ITERS \
    --run_name $name --headless > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $clip -> $name"
}
run walk2_subject1          4 faithful_walk2_s1   /tmp/train_faithful_walk2.log   &
run fight1_subject3         5 faithful_fight1_s3  /tmp/train_faithful_fight1s3.log &
run fallAndGetUp2_subject2  2 faithful_fall2_s2   /tmp/train_faithful_fall2s2.log &
wait
echo "ALL FAITHFUL SAMPLE TRAININGS DONE"
