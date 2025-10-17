#!/bin/bash
set -e # Exit immediately if a command fails

# --- Argument Parsing ---
if [ "$#" -ne 2 ]; then
    echo "Usage:   ./run_finetuning.sh <path_to_fasta> <model_size>"
    echo "Example: ./run_finetuning.sh my_sequences.fasta progen2-small"
    exit 1
fi

FASTA_FILE=$1
MODEL_SIZE=$2
# Get the base name of the fasta file (e.g., "my_sequences")
BASENAME=$(basename "${FASTA_FILE}" .fasta)

# Define dynamic directory names for organization
PREPARED_DATA_DIR="prepared_${BASENAME}"
OUTPUT_DIR="${MODEL_SIZE}-finetuned-on-${BASENAME}"

# --- Finetuning Workflow ---
echo "▶️ Starting finetuning workflow..."
echo "---------------------------------"
echo "Input FASTA:    ${FASTA_FILE}"
echo "Model Size:     ${MODEL_SIZE}"
echo "Data Output:    ${PREPARED_DATA_DIR}"
echo "Model Output:   ${OUTPUT_DIR}"
echo "---------------------------------"

# 1. Activate Virtual Environment
echo "▶️ Activating Python environment..."
source venv/bin/activate

# 2. Prepare Data
echo "▶️ Preparing data from ${FASTA_FILE}..."
python3 prepare_fasta.py --input_fasta "${FASTA_FILE}" --output_dir "${PREPARED_DATA_DIR}"

# 3. Run Finetuning
# It is recommended to run this inside a tmux session
echo "▶️ Launching finetuning script..."
python3 finetune_progen.py \
    --model="./checkpoints/${MODEL_SIZE}" \
    --train_file="${PREPARED_DATA_DIR}/train.txt" \
    --test_file="${PREPARED_DATA_DIR}/test.txt" \
    --output_dir="${OUTPUT_DIR}" \
    --use_lora

echo "✅ Finetuning complete. Best model saved in '${OUTPUT_DIR}/best_model/'"