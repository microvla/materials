#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

RUN_DATE="${RUN_DATE:-$(date +%m-%d)}"
export RUN_NAME_DATE="${RUN_NAME_DATE:-$(date +%m%d)}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${REPO_ROOT}/runs}"
RUN_ROOT_DIR="${RUN_ROOT_DIR%/}/${RUN_DATE}"
mkdir -p "${RUN_ROOT_DIR}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to the GPU indices allocated for this run." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-}"

# Every training option can be overridden by an environment variable. Keep the
# defaults here so the command printed in documentation and the launched config
# cannot silently diverge.
VLA_PATH="${VLA_PATH:-${REPO_ROOT}/checkpoints/openvla-7b}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-${REPO_ROOT}/MicroEnv/openvla_datasets}"
DATASET_NAME="${DATASET_NAME:-microenv_vessel_goal6}"
NUM_ACTIONS_CHUNK="${NUM_ACTIONS_CHUNK:-}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-1}"
USE_PROPRIO="${USE_PROPRIO:-True}"
USE_L1_REGRESSION="${USE_L1_REGRESSION:-True}"
USE_DIFFUSION="${USE_DIFFUSION:-False}"
USE_FILM="${USE_FILM:-False}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-100000}"
MAX_STEPS="${MAX_STEPS:-10005}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
SAVE_LATEST_CHECKPOINT_ONLY="${SAVE_LATEST_CHECKPOINT_ONLY:-False}"
IMAGE_AUG="${IMAGE_AUG:-False}"
USE_VAL_SET="${USE_VAL_SET:-True}"
VAL_FREQ="${VAL_FREQ:-2000}"
LORA_RANK="${LORA_RANK:-32}"
MERGE_LORA_DURING_TRAINING="${MERGE_LORA_DURING_TRAINING:-False}"
WANDB_PROJECT="${WANDB_PROJECT:-vla-micro}"
WANDB_ENTITY="${WANDB_ENTITY:-981498483-university-of-california-berkeley}"

DATASET_CONVERSION_METADATA="${DATA_ROOT_DIR}/${DATASET_NAME}/conversion_metadata.json"
if [[ -z "${NUM_ACTIONS_CHUNK}" && -f "${DATASET_CONVERSION_METADATA}" ]]; then
  NUM_ACTIONS_CHUNK="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["num_actions_chunk"])' "${DATASET_CONVERSION_METADATA}")"
fi
NUM_ACTIONS_CHUNK="${NUM_ACTIONS_CHUNK:-8}"
export NUM_ACTIONS_CHUNK

if [[ -z "${NPROC_PER_NODE}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    IFS=',' read -r -a _visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${#_visible_gpu_ids[@]}"
  else
    NPROC_PER_NODE="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  fi
fi

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "No visible CUDA devices detected. CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'" >&2
  exit 1
fi

echo "Launching OpenVLA MicroEnv finetune with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} and nproc_per_node=${NPROC_PER_NODE}"
echo "dataset=${DATASET_NAME} data_root=${DATA_ROOT_DIR} model=${VLA_PATH} action_chunk=${NUM_ACTIONS_CHUNK}"

python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --use_l1_regression "${USE_L1_REGRESSION}" \
  --use_diffusion "${USE_DIFFUSION}" \
  --use_film "${USE_FILM}" \
  --num_images_in_input "${NUM_IMAGES_IN_INPUT}" \
  --use_proprio "${USE_PROPRIO}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_steps_before_decay "${NUM_STEPS_BEFORE_DECAY}" \
  --max_steps "${MAX_STEPS}" \
  --save_freq "${SAVE_FREQ}" \
  --save_latest_checkpoint_only "${SAVE_LATEST_CHECKPOINT_ONLY}" \
  --image_aug "${IMAGE_AUG}" \
  --use_val_set "${USE_VAL_SET}" \
  --val_freq "${VAL_FREQ}" \
  --lora_rank "${LORA_RANK}" \
  --merge_lora_during_training "${MERGE_LORA_DURING_TRAINING}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}"
