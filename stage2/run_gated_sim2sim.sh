#!/bin/bash
# Sim2sim eval: gated VAE (EX_gated8) decoded vs original, tracked by the GATE-PASSING teachers.
# One gym.make per process. Survival + RMSE. GPU shared (eval is short).
set -u; cd /ws/user/yzdong/src/github/whole_body_tracking
VAE=${VAE:-$(for f in $(ls -t UniMoTok/experiments/biomechanics_tokenizer/${VAE_EXP:-EX_gated8}/checkpoints/epoch=*.ckpt); do [ "$(stat -c %s $f)" -gt 40000000 ] && echo "$f" && break; done)}
NORM=stage2/out/g1_dataset_T4within/normalization.npz
L=logs/rsl_rl/g1_flat; RUN=${VAE_EXP:-base}; G=${EVAL_GPU:-4}; LOG=${LOG:-stage2/out/gated_sim2sim.log}; : > $LOG
echo "VAE=$VAE" | tee -a $LOG
trunc(){ .venv/bin/python -c "
import numpy as np,sys
d=np.load(sys.argv[1],allow_pickle=True);a={k:d[k] for k in d.files};T=a['joint_pos'].shape[0];n=min(800,T)
for k,v in a.items():
 if hasattr(v,'shape') and getattr(v,'ndim',0)>=1 and v.shape[0]==T:a[k]=v[:n]
np.savez(sys.argv[2],**a)" "$1" "$2"; }
trk(){ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G OMNI_KIT_ACCEPT_EULA=YES OMNI_USER_HOME=/tmp/omni_gs2_${RUN}_$3 timeout 500 .venv/bin/python -u stage2/bench_earlyfreeze.py --motion "$1" --teacher_ckpt "$2" --no_freeze --late --steps 300 2>&1 | grep -oE "final survival = [0-9.]+" | grep -oE "[0-9.]+$"; }
clip(){ local c=$1 teach=$2 tag=$3
  local DS=stage2/out/g1_gs2_${RUN}_$tag; rm -rf $DS; mkdir -p $DS/val; cp $NORM $DS/normalization.npz
  local src=""; for s in train test val; do [ -f stage2/out/g1_dataset_T4/$s/$c.npz ] && src=stage2/out/g1_dataset_T4/$s/$c.npz && break; done
  cp "$src" $DS/val/$c.npz
  local rmse=$(.venv/bin/python stage2/sim2sim_vae_eval.py --phase01_only --skip_phase3 --vae_ckpt "$VAE" --dataset_dir $DS --splits val --clips $c --teacher_ckpt dummy --out $DS/p01.json 2>&1 | grep -oE "jt_rmse=[0-9.]+" | grep -oE "[0-9.]+")
  local dec=$DS/p01_decoded/${c}_decoded.npz
  trunc artifacts/$c:v0/motion.npz /tmp/gs2o_${RUN}_$tag.npz; trunc $dec /tmp/gs2d_${RUN}_$tag.npz
  local so=$(trk /tmp/gs2o_${RUN}_$tag.npz "$teach" o$tag); local sd=$(trk /tmp/gs2d_${RUN}_$tag.npz "$teach" d$tag)
  echo "$tag  rmse=$rmse  orig=$so  decoded=$sd" | tee -a $LOG
}
clip walk1_subject1                 $L/2026-05-30_15-59-30_walk_4090_full30k/model_29999.pt          walk
clip lafan_run1_subject2            $L/2026-06-05_12-28-59_lafan_run1_subject2_full30k/model_29999.pt run1
clip lafan_sprint1_subject2         $L/2026-06-05_17-32-20_sprint1_subj2_full30k/model_29999.pt       sprint1
clip lafan_dance1_subject1          $L/2026-06-01_21-02-09_lafan_suite_dance1_subject1/model_9999.pt   dance1
clip lafan_dance2_subject1          $L/2026-06-01_21-02-09_lafan_suite_dance2_subject1/model_9999.pt   dance2
clip lafan_fight1_subject2          $L/2026-06-01_21-02-09_lafan_suite_fight1_subject2/model_9999.pt   fight1
clip lafan_fightAndSports1_subject1 $L/2026-06-09_20-04-26_fightSports1_subj1_full30k_v2/model_29999.pt fightsports1
clip lafan_fallAndGetUp1_subject1   $L/2026-06-09_20-02-20_fallAndGetUp1_subj1_full30k_v2/model_29999.pt fallgetup
echo "GATED SIM2SIM DONE" | tee -a $LOG
