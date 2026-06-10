#!/bin/bash
# Phase B: higher-KL VAE sweep to resolve the gen-readiness (A) failure (latent not N(0,I)).
# Trains 3 KL values in parallel on GPUs 2/4/5 (UniMoTok = non-Isaac, parallel-safe).
# Run ONLY after Phase-A Isaac eval finishes (CPU contention slows Isaac).
cd /ws/user/yzdong/src/github/whole_body_tracking/UniMoTok
launch() {
  local kl=$1 gpu=$2
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu OMNI_KIT_ACCEPT_EULA=YES \
    nohup .venv_umt/bin/python -m training.train_tokenizer \
      --cfg /tmp/cfg_EX_T4w_kl$kl.yaml --nodebug > /tmp/train_kl$kl.log 2>&1 &
  echo "launched kl=$kl on GPU$gpu pid $!"
}
launch 1e-4 2
launch 1e-3 4
launch 1e-2 5
echo "KL sweep launched $(date +%H:%M:%S)"
