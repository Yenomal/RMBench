#!/usr/bin/env bash
#
# Planning Module pipeline: data preparation -> copy to LLaMA-Factory -> train -> merge LoRA.
# Edit only the variables in the "User configuration" section below. Do not edit source code.
#
# Usage:
#   cd policy/Mem-0
#   ./run_planning_pipeline.sh
#
# Optional: run specific steps (prepare, copy, train, merge):
#   STEPS="copy train merge" ./run_planning_pipeline.sh
#

set -e

# -----------------------------------------------------------------------------
# User configuration: edit only this section
# -----------------------------------------------------------------------------

# LeRobot dataset path (required). Example: /path/to/Mem-0/lerobot_datasets/battery_try
LEROBOT_DATASET_PATH=""

# Episode range for data preparation (inclusive start, exclusive end)
EPISODE_START_ID=0
EPISODE_END_ID=50

# LLaMA-Factory repository root (required). Example: /path/to/LlamaFactory
LLAMAFACTORY_ROOT=""

# Base directory for LoRA output and merged model (required). Script creates {dataset_name}_sft_lora under it.
BASE_OUTPUT_DIR=""

# Merged model output directory (optional). If empty, uses BASE_OUTPUT_DIR/Qwen3-VL-8B-Instruct-{dataset_name}
EXPORT_DIR=""

# Training options (optional; change if needed)
MAX_SAMPLES=1000
NUM_TRAIN_EPOCHS=25
PER_DEVICE_TRAIN_BATCH_SIZE=16
LEARNING_RATE="1.0e-4"
REPORT_TO="wandb"

# Merge options (optional)
EXPORT_SIZE=5
EXPORT_DEVICE="cpu"

# Conda environments: data prep uses CONDA_ENV_MEM0; train/merge use CONDA_ENV_LLAMAFACTORY
CONDA_ENV_MEM0="mem0"
CONDA_ENV_LLAMAFACTORY="llama_factory"

# Steps to run: prepare, copy, train, merge. Default: all. Example: STEPS="copy train merge"
STEPS="${STEPS:-prepare copy train merge}"

# -----------------------------------------------------------------------------
# Paths (do not edit unless you move the script)
# -----------------------------------------------------------------------------
MEM0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MEM0_DIR"
SCRIPTS_DIR="${MEM0_DIR}/scripts"
DATA_PREP_SCRIPT="${SCRIPTS_DIR}/llama_data_preparation/llamafactory_data_preparation.py"
PIPELINE_SCRIPT="${SCRIPTS_DIR}/planning_train_pipeline.py"

# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------
if [[ -z "$LEROBOT_DATASET_PATH" ]]; then
  echo "Error: LEROBOT_DATASET_PATH is not set. Edit the 'User configuration' section in this script." >&2
  exit 1
fi
if [[ ! -d "$LEROBOT_DATASET_PATH" ]]; then
  echo "Error: LEROBOT_DATASET_PATH is not a directory: $LEROBOT_DATASET_PATH" >&2
  exit 1
fi
if [[ -z "$LLAMAFACTORY_ROOT" ]]; then
  echo "Error: LLAMAFACTORY_ROOT is not set. Edit the 'User configuration' section in this script." >&2
  exit 1
fi
if [[ ! -d "$LLAMAFACTORY_ROOT" ]]; then
  echo "Error: LLAMAFACTORY_ROOT is not a directory: $LLAMAFACTORY_ROOT" >&2
  exit 1
fi
if [[ -z "$BASE_OUTPUT_DIR" ]]; then
  echo "Error: BASE_OUTPUT_DIR is not set. Edit the 'User configuration' section in this script." >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Step 1: Data preparation (runs in mem0 env)
# -----------------------------------------------------------------------------
if [[ " $STEPS " == *" prepare "* ]]; then
  echo "[Step 1] Data preparation (conda env: $CONDA_ENV_MEM0)..."
  conda run -n "$CONDA_ENV_MEM0" python "$DATA_PREP_SCRIPT" \
    --lerobot_dataset_path "$LEROBOT_DATASET_PATH" \
    --episode_start_id "$EPISODE_START_ID" \
    --episode_end_id "$EPISODE_END_ID"
  echo "[Step 1] Done."
fi

# -----------------------------------------------------------------------------
# Steps 2–4: Copy, train, merge (via Python pipeline script; train/merge use llama_factory env)
# -----------------------------------------------------------------------------
RUN_STEPS=""
for s in copy train merge; do
  if [[ " $STEPS " == *" $s "* ]]; then
    RUN_STEPS="${RUN_STEPS} ${s}"
  fi
done
RUN_STEPS="${RUN_STEPS# }"

if [[ -n "$RUN_STEPS" ]]; then
  echo "[Steps 2–4] Copy to LLaMA-Factory, train, merge (as specified: $RUN_STEPS)..."
  ARGS=(
    --lerobot_dataset_path "$LEROBOT_DATASET_PATH"
    --llamafactory_root "$LLAMAFACTORY_ROOT"
    --base_output_dir "$BASE_OUTPUT_DIR"
    --episode_start_id "$EPISODE_START_ID"
    --episode_end_id "$EPISODE_END_ID"
    --max_samples "$MAX_SAMPLES"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --learning_rate "$LEARNING_RATE"
    --report_to "$REPORT_TO"
    --export_size "$EXPORT_SIZE"
    --export_device "$EXPORT_DEVICE"
    --conda_env_mem0 "$CONDA_ENV_MEM0"
    --conda_env_llamafactory "$CONDA_ENV_LLAMAFACTORY"
    --steps $RUN_STEPS
  )
  if [[ -n "$EXPORT_DIR" ]]; then
    ARGS+=(--export_dir "$EXPORT_DIR")
  fi
  python "$PIPELINE_SCRIPT" "${ARGS[@]}"
  echo "[Steps 2–4] Done."
fi

echo "Pipeline finished."
