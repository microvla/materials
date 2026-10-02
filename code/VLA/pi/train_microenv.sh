#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to the GPU indices allocated for this run." >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Create the locked environment first: uv sync --locked" >&2
  exit 2
fi

config="${1:-pi05_microenv_lora}"
exp_name="${2:-${config}_reproduction}"
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

exec .venv/bin/python scripts/train.py "$config" \
  --exp-name "$exp_name" \
  --fsdp-devices "${#gpu_ids[@]}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --num-train-steps "${NUM_TRAIN_STEPS:-30000}" \
  --log-interval "${LOG_INTERVAL:-20}" \
  --save-interval "${SAVE_INTERVAL:-1000}" \
  --keep-period "${KEEP_PERIOD:-10000}"
