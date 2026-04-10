#!/bin/bash

# --- Configuration ---
# Set your config variables. All paths and parameters are defined here.
cd ${MTR_ROOT:-/path/to/mtr}/ablation_study/sharegpt
# Exit immediately on error
set -e

# Activate environment and set necessary env vars (These environment exports are necessary)
export NUMEXPR_MAX_THREADS=1000
source ${HOME_DIR:-/path/to}/.bashrc
mamba activate sglang
echo "HF_HOME is set to: $HF_HOME"
export WANDB_DISABLED=true
# 1. Dataset directory: containing sg_90k_part1.json sg_90k_part2.json files
# !!! Ensure this path is accessible in your environment !!!
export DATASET_PATH="${DATASETS_DIR:-/path/to/datasets}/liyucheng/ShareGPT90K"

# 2. Output file: Path to save detection results as JSONL
export OUTPUT_FILE="${MTR_ROOT:-/path/to/mtr}/ablation_study/sharegpt/result/topic_switch_results.jsonl"

# 3. Model name: Hugging Face Hubmodel identifier on Hub
export MODEL_NAME="Qwen/Qwen3-0.6B"
# MODEL_ID=""
# Qwen/Qwen3-235B-A22B-Instruct-2507
# 4. Max conversations (optional): For quick testingSet to empty string""or comment out to process all
# export MAX_CONVERSATIONS="100" # e.g., test with 100 conversations first

# --- Execution ---
# Below usually does not need modification

# Check if DATASET_PATH exists
if [ ! -d "$DATASET_PATH" ]; then
 echo "Error: Dataset path '$DATASET_PATH' does not exist or is not a directory"
    exit 1
fi

# Build command line arguments
CMD_ARGS="--dataset_path $DATASET_PATH \
          --output_file $OUTPUT_FILE \
          --model_name $MODEL_NAME"

# If MAX_CONVERSATIONS is set, add it to arguments
if [ -n "$MAX_CONVERSATIONS" ]; then
    CMD_ARGS="$CMD_ARGS --max_conversations $MAX_CONVERSATIONS"
fi

echo "=================================================="
echo "Starting topic switch detection..."
echo " - Model: $MODEL_NAME"
echo " - Dataset path: $DATASET_PATH"
echo " - Output file: $OUTPUT_FILE"
echo " - Conversations to process: ${MAX_CONVERSATIONS:-all}"
echo "=================================================="

# Execute Python script
# vLLM auto-uses all visible GPUs; tensor_parallel_size is set in Python
python topic_switch.py $CMD_ARGS

# Check if script executed successfully
if [ $? -eq 0 ]; then
 echo "=================================================="
 echo "Script executed successfully!"
 echo "Results saved to: $OUTPUT_FILE"
 echo "=================================================="
else
 echo "=================================================="
 echo "Script execution failed!"
 echo "=================================================="
fi