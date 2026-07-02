#!/bin/bash
# Periodically gate each KL run's latest FULL-SIZE checkpoint with the Phase-3 generative-readiness
# gate (no Isaac). Tracks reconstruction RMSE AND aggregated-posterior std vs KL — the KL sweep's
# whole question: does higher KL natively pull aggStd toward 1.0 (fixing Phase-3 A) and at what
# reconstruction cost? Logs to /tmp/kl_genready.log. Runs ~2.5h.
set -u
cd /ws/user/yzdong/src/github/whole_body_tracking
DS=stage2/out/g1_dataset_T4within
LOG=/tmp/kl_genready.log
echo "=== KL gen-readiness monitor started $(date +%H:%M:%S) ===" >> $LOG
for cycle in $(seq 1 22); do
  for kl in 1e-4 1e-3 1e-2; do
    dir=UniMoTok/experiments/biomechanics_tokenizer/EX_T4w_kl$kl/checkpoints
    # latest full-size epoch ckpt (avoid truncated last.ckpt / mid-write files)
    ck=$(ls -S $dir/epoch=*.ckpt 2>/dev/null | head -1)
    [ -z "$ck" ] && continue
    ep=$(grep -oE 'Epoch [0-9]+' /tmp/train_kl$kl.log 2>/dev/null | tail -1 | grep -oE '[0-9]+')
    timeout 400 .venv/bin/python stage2/sim2sim_vae_eval.py --phase01_only \
      --vae_ckpt "$ck" --dataset_dir "$DS" --splits val --teacher_ckpt dummy \
      --out /tmp/genready_kl$kl.json >/dev/null 2>&1
    res=$(.venv/bin/python -c "
import json
try:
  d=json.load(open('/tmp/genready_kl$kl.json'))
  g=d['gen_readiness']; a=g['A_prior_match']
  rmses=[c['phase0']['joint_angle_rmse_rad'] for c in d['clips'].values() if 'phase0' in c]
  rmse=sum(rmses)/len(rmses) if rmses else float('nan')
  print('aggStd=%.3f aggMu=%.3f active=%d/%d valRMSE=%.3f genReady=%s'%(a['agg_marginal_std'],a['agg_mu_mean_abs'],a['active_dims_kl>0.01'],a['latent_dim'],rmse,g['gen_ready']))
except Exception as e: print('parse-fail',e)
" 2>/dev/null)
    echo "$(date +%H:%M:%S) kl=$kl ep=${ep:-?} $res" >> $LOG
  done
  sleep 360
done
echo "=== monitor done $(date +%H:%M:%S) ===" >> $LOG
