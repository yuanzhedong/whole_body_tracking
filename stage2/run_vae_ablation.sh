#!/usr/bin/env bash
# VAE data-scaling ablation: train G1-VAE on 4 increasingly large motion subsets,
# run Phase-0 offline eval on all tiers, Phase-2 sim2sim on tiers that pass,
# and log a unified W&B report.
#
# Tiers:
#   T1  walk only                                      (baseline)
#   T2  walk + run + sprint                            (locomotion)
#   T3  T2 + dance1 + dance2                           (locomotion + rhythm)
#   T4  T3 + jumps + fallAndGetUp + fight + fightAndSports  (all categories)
#
# Usage:
#   nohup bash stage2/run_vae_ablation.sh > /tmp/vae_ablation_main.log 2>&1 &
set -euo pipefail

WBT=/ws/user/yzdong/src/github/whole_body_tracking
cd "$WBT"
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMNI_KIT_ACCEPT_EULA=YES

LOG="$WBT/stage2/out/vae_ablation.log"
mkdir -p "$WBT/stage2/out"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

WALK_CKPT="$WBT/logs/rsl_rl/g1_flat/2026-05-30_15-59-30_walk_4090_full30k/model_29999.pt"
ARTIFACTS="$WBT/artifacts"
OUT="$WBT/stage2/out"
PHASE0_RMSE_GATE=0.10

# ── Tier definitions ──────────────────────────────────────────────────────────
declare -A TIER_CLIPS
TIER_CLIPS[T1]="walk1_subject1"
TIER_CLIPS[T2]="walk1_subject1 lafan_run1_subject2 lafan_sprint1_subject2"
TIER_CLIPS[T3]="walk1_subject1 lafan_run1_subject2 lafan_sprint1_subject2 lafan_dance1_subject1 lafan_dance2_subject1"
TIER_CLIPS[T4]="walk1_subject1 lafan_run1_subject2 lafan_sprint1_subject2 lafan_dance1_subject1 lafan_dance2_subject1 lafan_jumps1_subject1 lafan_fallAndGetUp1_subject1 lafan_fight1_subject2 lafan_fightAndSports1_subject1"
TIERS=(T1 T2 T3 T4)

# Map clip name -> glob pattern to find its policy dir
# Format: "clip_name:glob_pattern"
declare -A CLIP_DIR_PATTERN
CLIP_DIR_PATTERN[lafan_jumps1_subject1]="*jumps1_subj1*"
CLIP_DIR_PATTERN[lafan_dance1_subject1]="*dance1_subj1*"
CLIP_DIR_PATTERN[lafan_dance2_subject1]="*dance2_subj1*"
CLIP_DIR_PATTERN[lafan_fallAndGetUp1_subject1]="*fallAndGetUp*"
CLIP_DIR_PATTERN[lafan_fight1_subject2]="*fight1_subj2*"
CLIP_DIR_PATTERN[lafan_fightAndSports1_subject1]="*fightSports*"
CLIP_DIR_PATTERN[lafan_run1_subject2]="*lafan_run1_subject2*"
CLIP_DIR_PATTERN[lafan_sprint1_subject2]="*sprint1*full30k*"
CLIP_DIR_PATTERN[walk1_subject1]="*walk_4090_full30k*"

wait_for_policy() {
    local clip=$1
    local pattern="${CLIP_DIR_PATTERN[$clip]:-*${clip}*}"
    log "  waiting: $clip (pattern: $pattern)"
    while ! ls "$WBT/logs/rsl_rl/g1_flat/"$pattern/model_*.pt > /dev/null 2>&1; do
        sleep 30
    done
    local ckpt
    ckpt=$(ls "$WBT/logs/rsl_rl/g1_flat/"$pattern/model_*.pt 2>/dev/null | sort -V | tail -1)
    log "  ready:   $clip -> $(basename $(dirname $ckpt)) ($(basename $ckpt))"
}

# ── Wait for T3+T4 policies (T1+T2 are already done) ─────────────────────────
log "=== VAE ABLATION START (fixed) ==="
log "T1+T2 policies already done. Waiting for dance1, dance2, jumps (T3)..."
for clip in lafan_dance1_subject1 lafan_dance2_subject1 lafan_jumps1_subject1; do
    wait_for_policy "$clip"
done
log "T3 clips ready. Exporting T1-T3 and starting VAE training..."

# T4 clips (fallAndGetUp, fight, fightAndSports) may finish later — wait separately
(
    log "  [T4 watcher] Waiting for fallAndGetUp, fight, fightAndSports..."
    for clip in lafan_fallAndGetUp1_subject1 lafan_fight1_subject2 lafan_fightAndSports1_subject1; do
        wait_for_policy "$clip"
    done
    log "  [T4 watcher] All T4 clips ready."
    # Export and train T4
    out_dir="$OUT/g1_dataset_T4"
    clips="${TIER_CLIPS[T4]}"
    log "  [T4] Exporting $out_dir..."
    .venv/bin/python "$WBT/stage2/export_g1_motion.py" \
        --artifacts_dir "$ARTIFACTS" --out_dir "$out_dir" \
        --target_fps 20 --to_yup --val_ratio 0.15 --test_ratio 0.15 \
        --include_clips $clips >> "$LOG" 2>&1
    log "  [T4] Export done. Training VAE on GPU 1..."
    export CUDA_VISIBLE_DEVICES=1
    export OMNI_USER_HOME=/tmp/omni_vae_T4
    cfg_tmp="/tmp/cfg_T4.yaml"
    sed "s|data_dir:.*g1_dataset.*|data_dir: ${out_dir}|g;
         s|name: g1_mldvae_v2_smallreg|name: g1_mldvae_T4|g;
         s|NAME: G1_MldVAE_v2_smallreg|NAME: G1_MldVAE_T4|g;
         s|FOLDER_EXP:.*|FOLDER_EXP: $WBT/UniMoTok/experiments/biomechanics_tokenizer/G1_MldVAE_T4|g;
         s|tags:.*|tags: ['mld_vae', 'g1_robot', 'ablation', 'T4']|g" \
        "$WBT/UniMoTok/configs/config_g1_mldvae_v2.yaml" > "$cfg_tmp"
    sed -i "s|data_dir: /ws.*g1_dataset.*|data_dir: ${out_dir}|g" "$cfg_tmp"
    cd "$WBT/UniMoTok"
    .venv_umt/bin/python -m training.train_tokenizer --cfg "$cfg_tmp" --nodebug \
        > "/tmp/vae_train_T4.log" 2>&1
    log "  [T4] VAE training done."
    cd "$WBT"
) &
T4_WATCHER_PID=$!
log "T4 watcher started (PID=$T4_WATCHER_PID)"

# ── Export T1, T2, T3 ─────────────────────────────────────────────────────────
for tier in T1 T2 T3; do
    clips="${TIER_CLIPS[$tier]}"
    out_dir="$OUT/g1_dataset_${tier}"
    log "Exporting $tier -> $out_dir  (clips: $clips)"
    .venv/bin/python "$WBT/stage2/export_g1_motion.py" \
        --artifacts_dir "$ARTIFACTS" --out_dir "$out_dir" \
        --target_fps 20 --to_yup --val_ratio 0.15 --test_ratio 0.15 \
        --include_clips $clips >> "$LOG" 2>&1
    log "Export $tier done."
done

# ── Train VAEs for T1, T2, T3 in parallel ────────────────────────────────────
log "Training VAEs for T1/T2/T3 in parallel (GPUs 2/4/1)..."
GPU_MAP=(T1:2 T2:4 T3:1)
VAE_PIDS=()
for mapping in "${GPU_MAP[@]}"; do
    tier="${mapping%%:*}"
    gpu="${mapping##*:}"
    out_dir="$OUT/g1_dataset_${tier}"
    exp_name="G1_MldVAE_${tier}"
    exp_dir="$WBT/UniMoTok/experiments/biomechanics_tokenizer/${exp_name}"
    cfg_tmp="/tmp/cfg_${tier}.yaml"
    sed "s|data_dir:.*g1_dataset.*|data_dir: ${out_dir}|g;
         s|name: g1_mldvae_v2_smallreg|name: g1_mldvae_${tier}|g;
         s|NAME: G1_MldVAE_v2_smallreg|NAME: ${exp_name}|g;
         s|FOLDER_EXP:.*|FOLDER_EXP: ${exp_dir}|g;
         s|tags:.*|tags: ['mld_vae', 'g1_robot', 'ablation', '${tier}']|g" \
        "$WBT/UniMoTok/configs/config_g1_mldvae_v2.yaml" > "$cfg_tmp"
    sed -i "s|data_dir: /ws.*g1_dataset.*|data_dir: ${out_dir}|g" "$cfg_tmp"
    log "  Starting VAE $tier on GPU $gpu..."
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        cd "$WBT/UniMoTok"
        .venv_umt/bin/python -m training.train_tokenizer --cfg "$cfg_tmp" --nodebug \
            > "/tmp/vae_train_${tier}.log" 2>&1
        log "  VAE $tier done."
    ) &
    VAE_PIDS+=($!)
done
wait "${VAE_PIDS[@]}"
log "T1/T2/T3 VAE training complete."

# ── Phase-0 eval for T1/T2/T3 ────────────────────────────────────────────────
log "Running Phase-0 eval for T1/T2/T3..."
PHASE0_RESULTS=()
for tier in T1 T2 T3; do
    exp_dir="$WBT/UniMoTok/experiments/biomechanics_tokenizer/G1_MldVAE_${tier}"
    ckpt=$(ls -t "$exp_dir/checkpoints/epoch=*.ckpt" 2>/dev/null | grep -v "\-v" | head -1 || echo "")
    [ -z "$ckpt" ] && log "  WARN: no ckpt for $tier" && continue
    out_json="$OUT/phase0_${tier}.json"
    dataset_dir="$OUT/g1_dataset_${tier}"
    log "  Phase-0 $tier: $ckpt"
    .venv/bin/python "$WBT/stage2/sim2sim_vae_eval.py" \
        --vae_ckpt "$ckpt" --dataset_dir "$dataset_dir" \
        --teacher_ckpt "$WALK_CKPT" --splits val \
        --phase01_only --out "$out_json" >> "$LOG" 2>&1 || true
    rmse=$(.venv/bin/python -c "import json; d=json.load(open('$out_json')); print(d.get('phase0_rmse_rad',d.get('joint_angle_rmse','?')))" 2>/dev/null || echo "?")
    log "  Phase-0 $tier: RMSE=$rmse rad"
    PHASE0_RESULTS+=("$tier:$rmse")
done

# ── Phase-2 sim2sim for passing tiers ────────────────────────────────────────
log "Phase-2 sim2sim for tiers with RMSE < $PHASE0_RMSE_GATE rad..."
export CUDA_VISIBLE_DEVICES=1
export OMNI_USER_HOME=/tmp/omni_sim2sim_ablation
rm -rf $OMNI_USER_HOME && mkdir -p $OMNI_USER_HOME
for tier in T1 T2 T3; do
    phase0_json="$OUT/phase0_${tier}.json"
    [ ! -f "$phase0_json" ] && continue
    rmse=$(.venv/bin/python -c "import json; d=json.load(open('$phase0_json')); print(d.get('phase0_rmse_rad',d.get('joint_angle_rmse','99')))" 2>/dev/null || echo "99")
    passes=$(.venv/bin/python -c "print('yes' if float('$rmse') < $PHASE0_RMSE_GATE else 'no')" 2>/dev/null || echo "no")
    if [ "$passes" = "yes" ]; then
        log "  $tier PASSES (RMSE=$rmse) -> sim2sim"
        exp_dir="$WBT/UniMoTok/experiments/biomechanics_tokenizer/G1_MldVAE_${tier}"
        ckpt=$(ls -t "$exp_dir/checkpoints/epoch=*.ckpt" 2>/dev/null | grep -v "\-v" | head -1)
        .venv/bin/python "$WBT/stage2/sim2sim_vae_eval.py" \
            --vae_ckpt "$ckpt" --dataset_dir "$OUT/g1_dataset_${tier}" \
            --teacher_ckpt "$WALK_CKPT" --splits val test \
            --eval_reps 2 --num_envs 128 \
            --out "$OUT/sim2sim_${tier}.json" \
            > "/tmp/sim2sim_${tier}.log" 2>&1 || true
        log "  sim2sim $tier done"
    else
        log "  $tier FAILS Phase-0 (RMSE=$rmse) -> skip sim2sim"
    fi
done

# ── W&B report ───────────────────────────────────────────────────────────────
log "Logging T1/T2/T3 ablation report to W&B..."
.venv/bin/python "$WBT/stage2/log_ablation_report.py" \
    --tiers T1 T2 T3 --out_dir "$OUT" >> "$LOG" 2>&1 || true

log "=== T1/T2/T3 ABLATION COMPLETE ==="
log "Results: ${PHASE0_RESULTS[*]}"
log "T4 still running in background (PID=$T4_WATCHER_PID)"
log "W&B: https://wandb.ai/cs224n-robustqa/g1-vae-ablation"

wait $T4_WATCHER_PID && log "T4 also complete."
