#!/bin/bash
# Phase 1: gate-check each LAFAN teacher's ORIGINAL survival (no-reset, 128 envs). >=0.95 = passes.
set -u; cd /ws/user/yzdong/src/github/whole_body_tracking
L=logs/rsl_rl/g1_flat; LOG=stage2/out/gate_check.log; : > $LOG
trunc(){ .venv/bin/python -c "
import numpy as np,sys
d=np.load(sys.argv[1],allow_pickle=True);a={k:d[k] for k in d.files};T=a['joint_pos'].shape[0];n=min(800,T)
for k,v in a.items():
 if hasattr(v,'shape') and getattr(v,'ndim',0)>=1 and v.shape[0]==T:a[k]=v[:n]
np.savez(sys.argv[2],**a)" "$1" "$2"; }
gate(){ local clip=$1 teach=$2
  [ -f "$teach" ] || { echo "$clip: NO TEACHER ($teach)" | tee -a $LOG; return; }
  trunc artifacts/$clip:v0/motion.npz /tmp/gc_$clip.npz
  s=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=YES OMNI_USER_HOME=/tmp/omni_gc timeout 400 .venv/bin/python -u stage2/bench_earlyfreeze.py --motion /tmp/gc_$clip.npz --teacher_ckpt "$teach" --no_freeze --late --steps 300 2>&1 | grep -oE "final survival = [0-9.]+" | grep -oE "[0-9.]+$")
  it=$(echo "$teach" | grep -oE "model_[0-9]+")
  echo "$clip  survival=$s  ($it)" | tee -a $LOG
}
gate walk1_subject1                 $L/2026-05-30_15-59-30_walk_4090_full30k/model_29999.pt
gate lafan_run1_subject2            $L/2026-06-05_12-28-59_lafan_run1_subject2_full30k/model_29999.pt
gate lafan_sprint1_subject2         $L/2026-06-05_17-32-20_sprint1_subj2_full30k/model_29999.pt
gate lafan_dance1_subject1          $L/2026-06-01_21-02-09_lafan_suite_dance1_subject1/model_9999.pt
gate lafan_dance2_subject1          $L/2026-06-01_21-02-09_lafan_suite_dance2_subject1/model_9999.pt
gate lafan_jumps1_subject1          $L/2026-06-06_09-25-46_jumps1_subj1_2h/model_6000.pt
gate lafan_fight1_subject2          $L/2026-06-01_21-02-09_lafan_suite_fight1_subject2/model_9999.pt
gate lafan_fightAndSports1_subject1 $L/2026-06-09_20-04-26_fightSports1_subj1_full30k_v2/model_29999.pt
gate lafan_fallAndGetUp1_subject1   $L/2026-06-09_20-02-20_fallAndGetUp1_subj1_full30k_v2/model_29999.pt
echo "GATE CHECK DONE" | tee -a $LOG
