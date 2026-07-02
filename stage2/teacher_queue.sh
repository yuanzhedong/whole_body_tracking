#!/bin/bash
# Self-refilling per-clip teacher queue: 4 GPU workers (1,2,4,5) atomically pop clips from
# /tmp/teacher_queue.txt and train until DRAINED — no manual relaunch between batches. Keeps all
# 4 GPUs hot autonomously. ensure()=csv_to_npz->registry if missing; train()=12k-iter tracking.
# Run via Bash run_in_background:true so the harness notifies on full drain -> launch next phase.
set -u; cd /ws/user/yzdong/src/github/whole_body_tracking
REG="cs224n-robustqa/wandb-registry-Motions"
ITERS=${ITERS:-12000}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
QUEUE=/tmp/teacher_queue.txt; LOCK=/tmp/teacher_queue.lock; CSV=/tmp/lafan1_dl/g1
LOG=/tmp/teacher_queue_run.log; : > "$LOG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$LOG"; }

pop(){ exec 9>"$LOCK"; flock 9; local c=$(head -n1 "$QUEUE" 2>/dev/null); [ -n "$c" ] && sed -i '1d' "$QUEUE"; flock -u 9; echo "$c"; }
in_registry(){ timeout 40 .venv/bin/python -c "import wandb;wandb.Api().artifact('$REG/lafan_$1:latest')" 2>/dev/null; }
ensure(){ local c=$1 g=$2
  in_registry "$c" && { log "$c in registry"; return 0; }
  log "convert $c (GPU$g)"
  WANDB_ENTITY=cs224n-robustqa CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g OMNI_KIT_ACCEPT_EULA=YES \
    OMNI_USER_HOME=/tmp/omni_cv_$c .venv/bin/python scripts/csv_to_npz.py --input_file $CSV/$c.csv \
    --input_fps 30 --output_name lafan_$c --headless > /tmp/cv_$c.log 2>&1 &
  local pid=$!
  for i in $(seq 1 45); do sleep 8; kill -0 $pid 2>/dev/null || break; in_registry "$c" && { kill -9 $pid 2>/dev/null; break; }; done
  pkill -9 -f "csv_to_npz.py.*$c.csv" 2>/dev/null; sleep 2; in_registry "$c"; }
train(){ local c=$1 g=$2
  log "TRAIN $c GPU$g ($ITERS it)"
  WANDB_ENTITY=cs224n-robustqa CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g OMNI_KIT_ACCEPT_EULA=YES \
    OMNI_USER_HOME=/tmp/omni_tr_$c .venv/bin/python scripts/rsl_rl/train.py --task Tracking-Flat-G1-v0 \
    --num_envs 2048 --registry_name "$REG/lafan_$c:latest" --max_iterations $ITERS --run_name teacher_$c \
    --headless > /tmp/train_teacher_$c.log 2>&1
  log "DONE $c"; }
worker(){ local g=$1; while true; do local c=$(pop); [ -z "$c" ] && break; ensure "$c" $g && train "$c" $g || log "SKIP $c"; done; log "WORKER GPU$g DONE"; }

log "QUEUE START: $(wc -l < $QUEUE) clips on GPU 1,2,4,5"
worker 1 & sleep 25; worker 2 & sleep 25; worker 4 & sleep 25; worker 5 &
wait
log "QUEUE DRAINED - ALL CLIPS DONE"
