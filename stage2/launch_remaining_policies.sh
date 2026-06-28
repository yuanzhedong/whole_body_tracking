#!/usr/bin/env bash
# Launch remaining 30k tracking policies for Path A (UniMoTok VAE data expansion).
# GPU 2 and 4 are free; first two clips start immediately, rest chain sequentially.
#
# Priority order (most diverse first):
#   GPU 2: lafan_jumps1_subject1 -> lafan_fallAndGetUp1_subject1 -> lafan_fight1_subject2
#   GPU 4: lafan_dance1_subject1 -> lafan_dance2_subject1 -> dance1_subject2 -> lafan_fightAndSports1_subject1

set -euo pipefail

WBT=/ws/user/yzdong/src/github/whole_body_tracking
cd "$WBT"

source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMNI_KIT_ACCEPT_EULA=YES
REGISTRY_BASE="cs224n-robustqa/wandb-registry-Motions"

LOG="$WBT/stage2/out/policy_launch.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_policy() {
    local clip="$1" gpu="$2" run_name="$3" log_file="$4"
    log "Starting $clip on GPU $gpu (run_name=$run_name)..."
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMNI_USER_HOME="/tmp/omni_${run_name}"
    rm -rf "$OMNI_USER_HOME" && mkdir -p "$OMNI_USER_HOME"
    .venv/bin/python "$WBT/scripts/rsl_rl/train.py" \
        --task Tracking-Flat-G1-v0 \
        --num_envs 2048 \
        --registry_name "$REGISTRY_BASE/$clip:latest" \
        --max_iterations 30000 \
        --run_name "$run_name" \
        --headless \
        > "$log_file" 2>&1
    log "Done: $clip (GPU $gpu)"
}

log "=== POLICY LAUNCH WATCHER STARTED ==="
log "GPU 2 chain: jumps -> fallAndGetUp -> fight"
log "GPU 4 chain: dance1_subj1 -> dance2_subj1 -> dance1_subj2 -> fightAndSports"

# GPU 2 chain (background)
(
    run_policy "lafan_jumps1_subject1"       2 "jumps1_subj1_full30k"       /tmp/train_jumps.log
    run_policy "lafan_fallAndGetUp1_subject1" 2 "fallAndGetUp1_subj1_full30k" /tmp/train_fall.log
    run_policy "lafan_fight1_subject2"        2 "fight1_subj2_full30k"        /tmp/train_fight.log
) &
GPU2_PID=$!
log "GPU 2 chain PID=$GPU2_PID"

# GPU 4 chain (background)
(
    run_policy "lafan_dance1_subject1"          4 "dance1_subj1_full30k"       /tmp/train_dance1_s1.log
    run_policy "lafan_dance2_subject1"          4 "dance2_subj1_full30k"       /tmp/train_dance2_s1.log
    run_policy "dance1_subject2"                4 "dance1_subj2_full30k"       /tmp/train_dance1_s2.log
    run_policy "lafan_fightAndSports1_subject1" 4 "fightSports1_subj1_full30k" /tmp/train_fightSports.log
) &
GPU4_PID=$!
log "GPU 4 chain PID=$GPU4_PID"

log "Both chains launched. Waiting for completion..."
wait "$GPU2_PID" && log "GPU 2 chain DONE"
wait "$GPU4_PID" && log "GPU 4 chain DONE"
log "=== ALL POLICIES DONE ==="
