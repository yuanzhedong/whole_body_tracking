#!/bin/bash
# Robust per-clip sim2sim validation: ONE gym.make per process (Isaac hangs on a 2nd gym.make
# in the same process). Per clip: build decoded npz (no Isaac), truncate orig+decoded to 800
# frames, then track each in a SEPARATE bench_earlyfreeze process (no-reset survival, LATE
# thresholds 0.25/0.8 = the normal termination thresholds). Survival = fraction never fell.
set -u
cd /ws/user/yzdong/src/github/whole_body_tracking
NORM=stage2/out/g1_dataset_T4within/normalization.npz
VAE=UniMoTok/experiments/biomechanics_tokenizer/EX_T4w_base/checkpoints/last.ckpt
L=logs/rsl_rl/g1_flat
GPU=1

trunc() {  # infile outfile N
  .venv/bin/python -c "
import numpy as np,sys
d=np.load(sys.argv[1],allow_pickle=True); a={k:d[k] for k in d.files}
T=a['joint_pos'].shape[0]; n=min(int(sys.argv[3]),T)
for k,v in a.items():
    if hasattr(v,'shape') and getattr(v,'ndim',0)>=1 and v.shape[0]==T: a[k]=v[:n]
np.savez(sys.argv[2],**a); print('trunc',sys.argv[1].split('/')[-1],T,'->',n)
" "$1" "$2" "$3"
}

track() {  # motion teacher  -> prints 'final survival'
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMNI_KIT_ACCEPT_EULA=YES \
    timeout 400 .venv/bin/python -u stage2/bench_earlyfreeze.py \
      --motion "$1" --teacher_ckpt "$2" --no_freeze --late --steps 300 2>&1 \
    | grep -E "final survival|failed=" | tail -2
}

run_clip() {  # feat art teacher tag
  local feat=$1 art=$2 teacher=$3 tag=$4
  echo "############ $tag ############ $(date +%H:%M:%S)"
  [ -f "$teacher" ] || { echo "SKIP $tag: no teacher $teacher"; return; }
  local DS=stage2/out/g1_simval_$tag
  rm -rf "$DS"; mkdir -p "$DS/val"; cp "$NORM" "$DS/normalization.npz"
  local src=""; for s in train test val; do [ -f stage2/out/g1_dataset_T4/$s/$feat.npz ] && src=stage2/out/g1_dataset_T4/$s/$feat.npz && break; done
  [ -z "$src" ] && { echo "SKIP $tag: no feat $feat"; return; }
  cp "$src" "$DS/val/$art.npz"
  echo "--- Phase0+1 (decode, no Isaac) ---"
  .venv/bin/python stage2/sim2sim_vae_eval.py --phase01_only --skip_phase3 \
    --vae_ckpt "$VAE" --dataset_dir "$DS" --splits val --clips "$art" --teacher_ckpt dummy \
    --out "$DS/p01.json" 2>&1 | grep -E "jt_rmse|saved"
  local dec="$DS/p01_decoded/${art}_decoded.npz"
  local orig="artifacts/$art:v0/motion.npz"
  [ -f "$dec" ] || { echo "SKIP $tag: no decoded npz at $dec"; return; }
  trunc "$orig" /tmp/orig_$tag.npz 800
  trunc "$dec"  /tmp/dec_$tag.npz  800
  echo "--- $tag ORIGINAL track ---"; track /tmp/orig_$tag.npz "$teacher"
  echo "--- $tag DECODED track ---";  track /tmp/dec_$tag.npz  "$teacher"
  echo "=== $tag end $(date +%H:%M:%S) ==="
}

run_clip walk1_subject1         walk1_subject1         $L/2026-05-30_15-59-30_walk_4090_full30k/model_29999.pt          walk
run_clip lafan_run1_subject2    lafan_run1_subject2    $L/2026-06-05_12-28-59_lafan_run1_subject2_full30k/model_29999.pt run1
run_clip lafan_sprint1_subject2 lafan_sprint1_subject2 $L/2026-06-05_17-32-20_sprint1_subj2_full30k/model_29999.pt       sprint1
run_clip lafan_dance1_subject1  lafan_dance1_subject1  $L/2026-06-01_21-02-09_lafan_suite_dance1_subject1/model_9999.pt   dance1
echo "ALL SIMVAL BENCH DONE $(date +%H:%M:%S)"
