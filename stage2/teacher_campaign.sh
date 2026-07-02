#!/bin/bash
# Per-clip tracking-policy (stage-1 teacher) CAMPAIGN: keep the 3 free 4090s (GPU 2,4,5) hot,
# consuming NEW diverse LAFAN1 clips. Each GPU worker chains: ensure-in-registry (csv_to_npz if
# missing) -> train ITERS iters. Instrumented stdout logs -> stage2/teacher_progress.py reads
# convergence. Caps iters to capture the gate for all tiers without burning to 30k.
set -u; cd /ws/user/yzdong/src/github/whole_body_tracking
REG="cs224n-robustqa/wandb-registry-Motions"
ITERS=${ITERS:-12000}
# thread caps: Isaac is GPU-bound; cap CPU thread pools to avoid oversubscription on the shared box
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
CSV=/tmp/lafan1_dl/g1

in_registry() { timeout 40 .venv/bin/python -c "import wandb;wandb.Api().artifact('$REG/lafan_$1:latest')" 2>/dev/null; }

ensure() { # clip gpu  -> convert+upload if not already in registry
  local c=$1 g=$2
  in_registry $c && { echo "[$(date +%H:%M:%S)] $c already in registry"; return 0; }
  echo "[$(date +%H:%M:%S)] converting $c on GPU$g"
  WANDB_ENTITY=cs224n-robustqa CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g OMNI_KIT_ACCEPT_EULA=YES \
    OMNI_USER_HOME=/tmp/omni_cv_$c .venv/bin/python scripts/csv_to_npz.py --input_file $CSV/$c.csv \
    --input_fps 30 --output_name lafan_$c --headless > /tmp/cv_$c.log 2>&1 &
  local pid=$!
  for i in $(seq 1 45); do
    sleep 8; kill -0 $pid 2>/dev/null || break
    in_registry $c && { echo "[$(date +%H:%M:%S)] $c uploaded (i=$i)"; kill -9 $pid 2>/dev/null; break; }
  done
  pkill -9 -f "csv_to_npz.py.*$c.csv" 2>/dev/null; sleep 2
  in_registry $c
}

train() { # clip gpu
  local c=$1 g=$2 name=teacher_$1 log=/tmp/train_teacher_$1.log
  echo "[$(date +%H:%M:%S)] TRAIN $c GPU$g ($ITERS it) -> $log"
  WANDB_ENTITY=cs224n-robustqa CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g OMNI_KIT_ACCEPT_EULA=YES \
    OMNI_USER_HOME=/tmp/omni_tr_$c .venv/bin/python scripts/rsl_rl/train.py --task Tracking-Flat-G1-v0 \
    --num_envs 2048 --registry_name "$REG/lafan_$c:latest" --max_iterations $ITERS --run_name $name \
    --headless > $log 2>&1
  echo "[$(date +%H:%M:%S)] DONE $c"
}

worker() { # gpu clip1 clip2 ...
  local g=$1; shift
  for c in "$@"; do
    ensure $c $g && train $c $g || echo "[$(date +%H:%M:%S)] SKIP $c (registry failed)"
  done
  echo "[$(date +%H:%M:%S)] WORKER GPU$g DONE"
}

# BATCH 2: 4 chains (GPU 1,2,4,5 all free), difficulty-balanced new clips. STAGGER ~25s to avoid the
# Isaac multi-proc startup race. None in registry yet -> ensure() converts each (csv_to_npz) first.
worker 1 fightAndSports1_subject4 dance1_subject3 walk4_subject1 &
sleep 25
worker 2 jumps1_subject5 dance2_subject3 walk2_subject3 &
sleep 25
worker 4 fallAndGetUp1_subject4 run1_subject5 dance2_subject4 &
sleep 25
worker 5 fallAndGetUp1_subject5 run2_subject4 walk1_subject5 &
wait
echo "CAMPAIGN COMPLETE"
