#!/bin/bash
# Compare sim2sim DECODED survival of two VAEs (gated23 vs laA) on representative gated clips, using
# each clip's gated teacher. Job queue "vae|clip" over 4 GPUs. orig survival = from gate_results.txt.
set -u; cd /ws/user/yzdong/src/github/whole_body_tracking
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
declare -A CK NORM
CK[gated23]=UniMoTok/experiments/biomechanics_tokenizer/EX_gated23_lat512/checkpoints/last.ckpt
CK[laA]=UniMoTok/experiments/biomechanics_tokenizer/EX_laA_lat512/checkpoints/last.ckpt
NORM[gated23]=stage2/out/g1_dataset_gated23
NORM[laA]=stage2/out/g1_dataset_lafan1_amass
RAW=stage2/out/lafan1_feats; RES=stage2/out/eval_gated_vae.txt; : > "$RES"
Q=/tmp/evae_q.txt; LK=/tmp/evae.lock
pop(){ exec 9>"$LK"; flock 9; local x=$(head -n1 "$Q" 2>/dev/null); [ -n "$x" ] && sed -i '1d' "$Q"; flock -u 9; echo "$x"; }
trunc(){ .venv/bin/python -c "
import numpy as np,sys
d=np.load(sys.argv[1],allow_pickle=True);a={k:d[k] for k in d.files};T=a['joint_pos'].shape[0];n=min(800,T)
for k,v in a.items():
 if hasattr(v,'shape') and getattr(v,'ndim',0)>=1 and v.shape[0]==T:a[k]=v[:n]
np.savez(sys.argv[2],**a)" "$1" "$2"; }
job(){ local vae=$1 clip=$2 g=$3 art=lafan_$clip
  local teach=$(ls -t logs/rsl_rl/g1_flat/*teacher_$clip/model_11999.pt 2>/dev/null | head -1)
  [ -z "$teach" ] && { echo "$vae $clip NO_TEACHER" >>"$RES"; return; }
  local DS=stage2/out/evae_${vae}_$clip; rm -rf $DS; mkdir -p $DS/val; cp ${NORM[$vae]}/normalization.npz $DS/
  cp $RAW/lafan1_$clip.npz $DS/val/$art.npz
  local rmse=$(.venv/bin/python stage2/sim2sim_vae_eval.py --phase01_only --skip_phase3 --vae_ckpt "${CK[$vae]}" \
    --dataset_dir $DS --splits val --clips $art --teacher_ckpt dummy --out $DS/p01.json 2>&1 | grep -oE "jt_rmse=[0-9.]+" | grep -oE "[0-9.]+")
  local dec=$DS/p01_decoded/${art}_decoded.npz
  [ -f "$dec" ] || { echo "$vae $clip NO_DECODE" >>"$RES"; return; }
  trunc "$dec" /tmp/evae_d_${vae}_$clip.npz
  local sd=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g OMNI_KIT_ACCEPT_EULA=YES OMNI_USER_HOME=/tmp/omni_ev_${vae}_$clip \
    timeout 500 .venv/bin/python -u stage2/bench_earlyfreeze.py --motion /tmp/evae_d_${vae}_$clip.npz --teacher_ckpt "$teach" \
    --no_freeze --late --steps 300 2>&1 | grep -oE "final survival = [0-9.]+" | grep -oE "[0-9.]+$")
  echo "$vae $clip rmse=${rmse:-NA} decoded_survival=${sd:-ERR}" | tee -a "$RES"
}
worker(){ local g=$1; while true; do local x=$(pop); [ -z "$x" ] && break; job "${x%%|*}" "${x##*|}" $g; done; }
# build queue: both VAEs x representative clips
: > $Q
for clip in walk1_subject2 walk3_subject1 dance1_subject2 fight1_subject3 jumps1_subject2; do
  echo "gated23|$clip" >> $Q; echo "laA|$clip" >> $Q
done
worker 1 & sleep 15; worker 2 & sleep 15; worker 4 & sleep 15; worker 5 &
wait
echo "=== RESULTS (orig survival from gate_results.txt) ==="; sort "$RES"
