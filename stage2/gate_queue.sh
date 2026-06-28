#!/bin/bash
# Gate-check queue: for each trained teacher, sim2sim survival on its ORIGINAL motion (no VAE).
# PASS = survival >= 0.95 (the "good teacher" gate). 4 GPU workers pop from /tmp/gate_queue.txt.
# Truncates motion to 800 frames (avoids slow gym.make on long clips). Results -> stage2/out/gate_results.txt
set -u; cd /ws/user/yzdong/src/github/whole_body_tracking
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
QUEUE=/tmp/gate_queue.txt; LOCK=/tmp/gate_queue.lock; RES=stage2/out/gate_results.txt; : > "$RES"
pop(){ exec 9>"$LOCK"; flock 9; local c=$(head -n1 "$QUEUE" 2>/dev/null); [ -n "$c" ] && sed -i '1d' "$QUEUE"; flock -u 9; echo "$c"; }
trunc(){ .venv/bin/python -c "
import numpy as np,sys
d=np.load(sys.argv[1],allow_pickle=True);a={k:d[k] for k in d.files};T=a['joint_pos'].shape[0];n=min(800,T)
for k,v in a.items():
 if hasattr(v,'shape') and getattr(v,'ndim',0)>=1 and v.shape[0]==T:a[k]=v[:n]
np.savez(sys.argv[2],**a)" "$1" "$2"; }
gate(){ local c=$1 g=$2
  local ck=$(ls -t logs/rsl_rl/g1_flat/*teacher_$c/model_11999.pt 2>/dev/null | head -1)
  local mo=artifacts/lafan_$c:v0/motion.npz
  [ -z "$ck" ] && { echo "$c NO_CKPT" >> "$RES"; return; }
  [ -f "$mo" ] || { echo "$c NO_MOTION" >> "$RES"; return; }
  trunc "$mo" /tmp/gate_o_$c.npz
  local s=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g OMNI_KIT_ACCEPT_EULA=YES OMNI_USER_HOME=/tmp/omni_gate_$c \
    timeout 500 .venv/bin/python -u stage2/bench_earlyfreeze.py --motion /tmp/gate_o_$c.npz --teacher_ckpt "$ck" \
    --no_freeze --late --steps 300 2>&1 | grep -oE "final survival = [0-9.]+" | grep -oE "[0-9.]+$")
  local pass=$(.venv/bin/python -c "print('PASS' if float('${s:-0}')>=0.95 else 'fail')" 2>/dev/null)
  echo "$c survival=${s:-ERR} $pass" >> "$RES"; echo "[$(date +%H:%M:%S)] $c survival=${s:-ERR} $pass"
}
worker(){ local g=$1; while true; do local c=$(pop); [ -z "$c" ] && break; gate "$c" $g; done; echo "[gate] GPU$g done"; }
worker 1 & sleep 15; worker 2 & sleep 15; worker 4 & sleep 15; worker 5 &
wait
echo "GATE-CHECK DONE -> $RES"
sort -t= -k2 -rn "$RES"
