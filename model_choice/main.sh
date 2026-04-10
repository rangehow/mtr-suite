#!/bin/bash

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
# input_dir="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/mtr"

input_dir="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/finance_unfiltered"
output_dir="${MTR_ROOT:-/path/to/mtr}/tempfile/finance_choice"
cache_dir="${MTR_ROOT:-/path/to/mtr}/tempfile/finance_unfiltered_cache"

# input_dir="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/finance_filtered"
# output_dir="${MTR_ROOT:-/path/to/mtr}/tempfile/finance_filtered_choice"
# cache_dir="${MTR_ROOT:-/path/to/mtr}/tempfile/finance_filtered_cache"

inference_backend='vllm'

judge_model_name="$1"
turn="$2"
start="$3"
end="$4"
tested_model_name="$5"

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


# Check arguments
if [ -z "$judge_model_name" ] || [ -z "$start" ] || [ -z "$end" ] || [ -z "$turn" ]; then
 echo "Error: Please provide model name, start, end, and turn as arguments"
 echo "Available model names include"
  while IFS= read -r model_name; do
 echo "- $model_name"
  done < <(list_available_models) # Using process substitution to read model list
  exit 1
fi

# Get model path
model_path=$(get_model_path "$judge_model_name" "$CONFIG_FILE")

# Check if model path was found
if [ -n "$model_path" ]; then
 echo "Using model: $judge_model_name"
 echo "Model path: $model_path"

 # Execute Python script
  python ${MTR_ROOT:-/path/to/mtr}/model_choice.py \
    --tested_model_name "$tested_model_name" \
    --model_path "$model_path" \
    --output_dir "$output_dir" \
    --input_dir "$input_dir" \
    --start "$start" \
    --end "$end" \
    --turn "$turn" \
    --inference_backend "$inference_backend" \
    --judge_model_name $judge_model_name \

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