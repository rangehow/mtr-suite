#!/bin/bash
cd ${MTR_ROOT:-/path/to/mtr}/analysis_of_previous_benchmark
# Load environment (unchanged)
source ${HOME_DIR:-/path/to}/.bashrc
mamba activate sglang

set -e

# --- Load model info from config file ---
# Assumes config file is in the same directory as script, or provide full path
CONFIG_FILE="models.conf"
if [ ! -f "$CONFIG_FILE" ]; then
 echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Other variables (unchanged)
output_dir="${MTR_ROOT:-/path/to/mtr}/tempfile/chatrag"
inference_backend='vllm'
chatrag_benchmark_dir='${DATASETS_DIR:-/path/to/datasets}/nvidia/ChatRAG-Bench'
coral_dir="${HOME_DIR:-/path/to}/datasets/new_coral_hf"

judge_model_name="$1"
target="$2"
# --- Helper function to list models from config ---
list_available_models() {
 # Ignore empty lines and comments, extract content before = sign
  grep -v '^[[:space:]]*$' "$CONFIG_FILE" | grep -v '^[[:space:]]*#' | cut -d'=' -f1
}

# --- Helper function to get model path from config ---
# $1: model name
# $2: config file path
get_model_path() {
  local name="$1"
  local file="$2"
 # Use grep to find lines starting with "name=" (ensure exact match)
 # Use head -n 1 to prevent duplicate configs (take first)
 # Use cut to extract everything after = sign (path)
  local line=$(grep "^${name}=" "$file" | head -n 1)
  if [ -n "$line" ]; then
 # Extract everything after the first = sign
 echo "${line#*=}"
  else
 echo "" # Return empty string if not found
  fi
}



# Get model path
model_path=$(get_model_path "$judge_model_name" "$CONFIG_FILE")

# Check if model path was found
if [ -n "$model_path" ]; then
 echo "Using model: $judge_model_name"
 echo "Model path: $model_path"

 # Execute Python script
  python ${MTR_ROOT:-/path/to/mtr}/analysis_of_previous_benchmark.py \
    --model_path "$model_path" \
    --output_dir "$output_dir" \
    --inference_backend "$inference_backend" \
    --judge_model_name $judge_model_name \
    --target $target \
    --chatrag-bench-dir $chatrag_benchmark_dir\
    --coral-dir $coral_dir

else
 # If model path not found
 echo "Error: No model named '$judge_model_name' found"
 echo "Available model names include"
  while IFS= read -r model_name; do
 echo "- $model_name"
  done < <(list_available_models)
  exit 1
fi

echo "Script execution complete."