#!/bin/bash
# Gate the latest EX_T4w_hardup checkpoint every ~8 min; log per-clip Phase-0 RMSE so we can
# see whether up-weighting drops the HARD clips' reconstruction error. (RMSE is an early proxy;
# the real test is the sim2sim re-validation once converged.)
cd /ws/user/yzdong/src/github/whole_body_tracking
LOG=/tmp/hardup_rmse.log
DS=stage2/out/g1_dataset_T4within   # gate on the honest within-clip val split
for cyc in $(seq 1 30); do
  ck=$(ls -S UniMoTok/experiments/biomechanics_tokenizer/EX_T4w_hardup/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
  ep=$(grep -oE 'Epoch [0-9]+' /tmp/train_hardup.log 2>/dev/null | tail -1 | grep -oE '[0-9]+')
  if [ -n "$ck" ]; then
    timeout 400 .venv/bin/python stage2/sim2sim_vae_eval.py --phase01_only --skip_phase3 \
      --vae_ckpt "$ck" --dataset_dir "$DS" --splits val --teacher_ckpt dummy \
      --out /tmp/hardup_p0.json >/tmp/hardup_p0.out 2>&1
    line=$(grep -E "jt_rmse" /tmp/hardup_p0.out | sed -E 's/.*\] +([a-z0-9_]+) .*jt_rmse=([0-9.]+).*/\1=\2/' | tr '\n' ' ')
    echo "$(date +%H:%M:%S) ep=${ep:-?} $line" >> $LOG
  fi
  sleep 480
done
