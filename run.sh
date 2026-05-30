#!/usr/bin/env bash
# Wrapper: load .env, activate the venv, and run a python script with the
# correct Isaac Sim / WandB environment.
#
#   ./run.sh scripts/csv_to_npz.py --input_file motions/dance1_subject2.csv ...
#   ./run.sh scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 ...
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Load .env (export every variable defined in it)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Warn if WandB isn't configured (the stock scripts need it)
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[run.sh] WARNING: WANDB_API_KEY is empty — train.py/csv_to_npz.py will fail to reach the registry." >&2
fi

exec python "$@"
