#!/bin/bash
# Continuous AMASS per-clip teacher queue: 4 GPU workers churn through /tmp/amass_queue.txt (HF g1/*.npy
# paths) self-refilling & autonomous. Per clip: download .npy -> CSV(+0.793 z) -> csv_to_npz->registry ->
# train 12k -> SAVE model to W&B (cs224n-robustqa/g1-teachers) -> DELETE local run+files (disk hygiene).
# Stoppable anytime (pkill -f amass_teacher_queue). Progress in /tmp/amass_teacher_queue.log.
set -u; cd /ws/user/yzdong/src/github/whole_body_tracking
REG="cs224n-robustqa/wandb-registry-Motions"; ITERS=${ITERS:-12000}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export WANDB_ENTITY=cs224n-robustqa CUDA_DEVICE_ORDER=PCI_BUS_ID OMNI_KIT_ACCEPT_EULA=YES
Q=/tmp/amass_queue.txt; LK=/tmp/amass_q.lock; LOG=/tmp/amass_teacher_queue.log
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$LOG"; }
pop(){ exec 9>"$LK"; flock 9; local x=$(head -n1 "$Q" 2>/dev/null); [ -n "$x" ] && sed -i '1d' "$Q"; flock -u 9; echo "$x"; }
in_reg(){ timeout 40 .venv/bin/python -c "import wandb;wandb.Api().artifact('$REG/$1:latest')" 2>/dev/null; }

do_clip(){ local hf=$1 g=$2
  local base=$(basename "$hf" .npy); local fps=$(echo "$base" | grep -oE "_[0-9]+_jpos" | grep -oE "[0-9]+")
  local name="amass_$(echo "$hf" | sed 's#g1/##;s#/#_#g;s#.npy##' | tr ' ' '_' | cut -c1-72)"
  local csv=/tmp/at_$g.csv npy=/tmp/at_$g.npy
  # download + convert to CSV (+0.793 z)
  if ! in_reg "$name"; then
    .venv/bin/python - "$hf" "$npy" "$csv" <<'PY' 2>>/tmp/at_dl_$g.log || { log "DL/CONV FAIL $name"; return; }
import sys,numpy as np
from huggingface_hub import hf_hub_download
hf,npy,csv=sys.argv[1:4]
lp=hf_hub_download('fleaven/Retargeted_AMASS_for_robotics',hf,repo_type='dataset',local_dir='/tmp/amass_dl_q')
d=np.load(lp).astype(np.float64)
if d.ndim!=2 or d.shape[1]<36 or len(d)<8: sys.exit(1)
d[:,2]+=0.793; np.savetxt(csv,d,delimiter=',',fmt='%.6f')
PY
    OMNI_USER_HOME=/tmp/omni_ac_$g CUDA_VISIBLE_DEVICES=$g timeout 300 .venv/bin/python scripts/csv_to_npz.py \
      --input_file "$csv" --input_fps "${fps:-30}" --output_name "$name" --headless >/tmp/at_conv_$g.log 2>&1 &
    local cp=$!; for i in $(seq 1 35); do sleep 8; kill -0 $cp 2>/dev/null||break; in_reg "$name" && { kill -9 $cp 2>/dev/null; break; }; done
    pkill -9 -f "csv_to_npz.py.*$csv" 2>/dev/null; sleep 1
    in_reg "$name" || { log "REGISTER FAIL $name"; return; }
  fi
  # train teacher
  local rn=teacher_$name
  OMNI_USER_HOME=/tmp/omni_at_$g CUDA_VISIBLE_DEVICES=$g .venv/bin/python scripts/rsl_rl/train.py \
    --task Tracking-Flat-G1-v0 --num_envs 2048 --registry_name "$REG/$name:latest" \
    --max_iterations $ITERS --run_name "$rn" --headless >/tmp/at_train_$g.log 2>&1
  local rd=$(ls -dt logs/rsl_rl/g1_flat/*_$rn 2>/dev/null | head -1)
  [ -z "$rd" ] || [ ! -f "$rd/model_$((ITERS-1)).pt" ] && { log "TRAIN FAIL $name"; return; }
  # save to W&B + cleanup local
  .venv/bin/python stage2/save_one_teacher.py "$name" "$rd" "$((ITERS-1))" >/dev/null 2>&1 \
    && { log "SAVED+DONE $name"; rm -rf "$rd"; } || log "WANDB-SAVE FAIL $name (kept local $rd)"
  rm -f "$csv" "$npy"; rm -rf /tmp/amass_dl_q/$(dirname "$hf") 2>/dev/null
}
worker(){ local g=$1; while true; do local hf=$(pop); [ -z "$hf" ] && break; do_clip "$hf" $g; done; log "WORKER GPU$g DONE"; }
log "AMASS QUEUE START: $(wc -l < $Q) clips on GPU 1,2,4,5"
worker 1 & sleep 25; worker 2 & sleep 25; worker 4 & sleep 25; worker 5 &
wait; log "AMASS QUEUE DRAINED"
