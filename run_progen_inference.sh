#!/bin/bash
set -e # Exit immediately if a command fails

# --- Argument Parsing ---
if [ "$#" -lt 2 ]; then
    echo "Usage:   ./run_inference.sh <base_model_size> <path_to_lora_adapter>"
    echo "Example: ./run_inference.sh progen2-small progen2-small-finetuned-on-your_data/best_model"
    exit 1
fi

BASE_MODEL_SIZE=$1
LORA_ADAPTER_PATH=$2
# Use the third argument for num_samples, or default to 5 if not provided
NUM_SAMPLES=${3:-5}

# --- Inference Workflow ---
echo "▶️ Generating ${NUM_SAMPLES} sequences..."
echo "---------------------------------"
echo "Base Model:     ${BASE_MODEL_SIZE}"
echo "LoRA Adapter:   ${LORA_ADAPTER_PATH}"
echo "---------------------------------"

# 1. Activate Virtual Environment
echo "▶️ Activating Python environment..."
source venv/bin/activate

# 2. Run Inference Script
python3 sample.py \
    --base_model_path="./checkpoints/${BASE_MODEL_SIZE}" \
    --lora_adapter_path="${LORA_ADAPTER_PATH}" \
    --num_samples=${NUM_SAMPLES}

echo "✅ Inference complete."
