#!/bin/bash

# This script downloads and configures a specified ProGen2 model.
# Usage: ./setup_model.sh <model_name>
# Example: ./setup_model.sh progen2-medium

set -e # Exit immediately if a command fails

MODEL_NAME=$1

# Check if a model name was provided
if [ -z "$MODEL_NAME" ]; then
  echo "Error: No model name provided."
  echo "Usage: ./setup_model.sh <model_name>"
  echo "Example: ./setup_model.sh progen2-medium"
  exit 1
fi

# Check if the checkpoint directory already exists
if [ -d "checkpoints/$MODEL_NAME" ]; then
  echo "✅ Checkpoint for $MODEL_NAME already exists. Nothing to do."
  exit 0
fi

echo "--- Setting up model: $MODEL_NAME ---"

# 1. Download the official model weights from Google
echo "➡️ Downloading model weights..."
wget "https://storage.googleapis.com/sfr-progen-research/checkpoints/${MODEL_NAME}.tar.gz"

# 2. Create the checkpoint directory and extract the weights
echo "➡️ Extracting files..."
mkdir -p "checkpoints/${MODEL_NAME}"
tar -xvf "${MODEL_NAME}.tar.gz" -C "checkpoints/${MODEL_NAME}/"

# 3. Check for and copy the generic tokenizer file
TOKENIZER_SOURCE_PATH="./progen/progen2/tokenizer.json"
if [ ! -f "checkpoints/$MODEL_NAME/tokenizer.json" ]; then
    if [ -f "$TOKENIZER_SOURCE_PATH" ]; then
        echo "➡️ Copying tokenizer.json..."
        cp "$TOKENIZER_SOURCE_PATH" "checkpoints/${MODEL_NAME}/"
    else
        # Fallback if the progen repo was cleaned up
        echo "❗️ tokenizer.json not found in progen repo. Cloning to get it..."
        git clone https://github.com/salesforce/progen.git temp_progen_repo
        cp ./temp_progen_repo/progen2/tokenizer.json "checkpoints/${MODEL_NAME}/"
        rm -rf temp_progen_repo
    fi
fi

# 4. Clean up the downloaded archive
echo "➡️ Cleaning up..."
rm "${MODEL_NAME}.tar.gz"

echo "✅ Setup complete for $MODEL_NAME!"