#!/bin/bash
# Re-validate hard clips (+walk regression) with the UP-WEIGHTED VAE vs EX_T4w_base baselines,
# using the SAME (best available) teachers. One gym.make per process.
set -u
cd /ws/user/yzdong/src/github/whole_body_tracking
NORM=stage2/out/g1_dataset_T4within/normalization.npz
VAE="UniMoTok/experiments/biomechanics_tokenizer/EX_T4w_hardup/checkpoints/epoch=1020.ckpt"
L=logs/rsl_rl/g1_flat
trunc(){ .venv/bin/python -c "
import numpy as np,sys
d=np.load(sys.argv[1],allow_pickle=True);a={k:d[k] for k in d.files}
T=a['joint_pos'].shape[0];n=min(int(sys.argv[3]),T)
for k,v in a.items():
 if hasattr(v,'shape') and getattr(v,'ndim',0)>=1 and v.shape[0]==T:a[k]=v[:n]
np.savez(sys.argv[2],**a)" "$1" "$2" "$3"; }
track(){ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=YES OMNI_USER_HOME=/tmp/omni_reval2 timeout 400 .venv/bin/python -u stage2/bench_earlyfreeze.py --motion "$1" --teacher_ckpt "$2" --no_freeze --late --steps 300 2>&1 | grep -E "final survival"; }
run_clip(){ local feat=$1 art=$2 teacher=$3 tag=$4
 echo "######## $tag ######## $(date +%H:%M:%S)"
 [ -f "$teacher" ]||{ echo "SKIP $tag no teacher";return;}
 local DS=stage2/out/g1_reval_$tag; rm -rf "$DS"; mkdir -p "$DS/val"; cp "$NORM" "$DS/normalization.npz"
 local src=""; for s in train test val; do [ -f stage2/out/g1_dataset_T4/$s/$feat.npz ]&&src=stage2/out/g1_dataset_T4/$s/$feat.npz&&break;done
 [ -z "$src" ]&&{ echo "SKIP $tag no feat";return;}; cp "$src" "$DS/val/$art.npz"
 .venv/bin/python stage2/sim2sim_vae_eval.py --phase01_only --skip_phase3 --vae_ckpt "$VAE" --dataset_dir "$DS" --splits val --clips "$art" --teacher_ckpt dummy --out "$DS/p01.json" 2>&1 | grep -E "jt_rmse"
 local dec="$DS/p01_decoded/${art}_decoded.npz"; local orig="artifacts/$art:v0/motion.npz"
 [ -f "$dec" ]||{ echo "SKIP $tag no decoded";return;}
 trunc "$orig" /tmp/ro_$tag.npz 800; trunc "$dec" /tmp/rd_$tag.npz 800
 echo -n "  ORIGINAL: "; track /tmp/ro_$tag.npz "$teacher"
 echo -n "  DECODED:  "; track /tmp/rd_$tag.npz "$teacher"
}
run_clip lafan_fallAndGetUp1_subject1   lafan_fallAndGetUp1_subject1   $L/2026-06-09_20-02-20_fallAndGetUp1_subj1_full30k_v2/model_29999.pt fallgetup
run_clip lafan_fightAndSports1_subject1 lafan_fightAndSports1_subject1 $L/2026-06-09_20-04-26_fightSports1_subj1_full30k_v2/model_29999.pt fightsports1
run_clip lafan_sprint1_subject2         lafan_sprint1_subject2         $L/2026-06-05_17-32-20_sprint1_subj2_full30k/model_29999.pt sprint1
run_clip lafan_fight1_subject2          lafan_fight1_subject2          $L/2026-06-01_21-02-09_lafan_suite_fight1_subject2/model_9999.pt fight1
run_clip walk1_subject1                 walk1_subject1                 $L/2026-05-30_15-59-30_walk_4090_full30k/model_29999.pt walk
echo "REVAL DONE $(date +%H:%M:%S)"
