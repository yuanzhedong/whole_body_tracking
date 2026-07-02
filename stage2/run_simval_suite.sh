#!/bin/bash
# Autonomous per-clip sim2sim validation suite (serial — Isaac is a machine-wide singleton).
# Each clip: build a temp full-clip eval dataset (T4within normalization + full clip features),
# run sim2sim_vae_eval with that clip's trained teacher, no-reset survival, unbuffered.
set -u
cd /ws/user/yzdong/src/github/whole_body_tracking
NORM=stage2/out/g1_dataset_T4within/normalization.npz
VAE=UniMoTok/experiments/biomechanics_tokenizer/EX_T4w_base/checkpoints/last.ckpt
L=logs/rsl_rl/g1_flat

run_one() {
  local feat=$1 art=$2 teacher=$3 tag=$4
  local DS=stage2/out/g1_simval_$tag
  echo "============================================================"
  echo "=== SIMVAL $tag  feat=$feat art=$art  $(date +%H:%M:%S) ==="
  if [ ! -f "$teacher" ]; then echo "SKIP $tag: teacher missing $teacher"; return; fi
  rm -rf "$DS"; mkdir -p "$DS/val"; cp "$NORM" "$DS/normalization.npz"
  local src=""
  for s in train test val; do
    [ -f "stage2/out/g1_dataset_T4/$s/$feat.npz" ] && src="stage2/out/g1_dataset_T4/$s/$feat.npz" && break
  done
  if [ -z "$src" ]; then echo "SKIP $tag: no feature clip $feat.npz"; return; fi
  cp "$src" "$DS/val/$art.npz"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
    timeout 900 .venv/bin/python -u stage2/sim2sim_vae_eval.py \
      --vae_ckpt "$VAE" --dataset_dir "$DS" --splits val --clips "$art" \
      --teacher_ckpt "$teacher" --skip_phase3 --max_steps 300 --eval_reps 1 \
      --out "stage2/out/simval_$tag.json" 2>&1 | grep -vE "Warning|carb|omni\.|\[Info\]|P2P|D\\\\D|Bus-ID|GPU |Driver|^\| |Vendor|Memory|Cores|Kit"
  echo "=== $tag exit=${PIPESTATUS[0]} $(date +%H:%M:%S) ==="
}

run_one walk1_subject1        walk1_subject1         $L/2026-05-30_15-59-30_walk_4090_full30k/model_29999.pt        walk
run_one lafan_run1_subject2   lafan_run1_subject2    $L/2026-06-05_12-28-59_lafan_run1_subject2_full30k/model_29999.pt run1
run_one lafan_sprint1_subject2 lafan_sprint1_subject2 $L/2026-06-05_17-32-20_sprint1_subj2_full30k/model_29999.pt    sprint1
run_one lafan_dance1_subject1 lafan_dance1_subject1  $L/2026-06-01_21-02-09_lafan_suite_dance1_subject1/model_9999.pt dance1
echo "ALL SIMVAL DONE $(date +%H:%M:%S)"
