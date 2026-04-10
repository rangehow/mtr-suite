#!/bin/bash

# Load environment (unchanged)
source ${HOME_DIR:-/path/to}/.bashrc
mamba activate sglang


# --- Load model info from config file ---
# Assumes config file is in the same directory as script, or provide full path
CONFIG_FILE="models.conf"
if [ ! -f "$CONFIG_FILE" ]; then
 echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Other variables (unchanged)
dataset_path="$6"
output_dir="$7"
cache_dir="$8"
inference_backend='vllm'
query_model_name="$1"
turn="$2"
start="$3"
end="$4"
response_model_name="$5"
last_turn_dataset="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/mtr/Llama-4-Scout-17B-16E-Instruct-Llama-4-Scout-17B-16E-Instruct/1/0-999"
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
if [ -z "$query_model_name" ] || [ -z "$start" ] || [ -z "$end" ] || [ -z "$turn" ]; then
 echo "Error: Please provide model name, start, end, and turn as arguments"
 echo "Available model names include"
  while IFS= read -r model_name; do
 echo "- $model_name"
  done < <(list_available_models) # Using process substitution to read model list
  exit 1
fi

# Get model path
query_model_path=$(get_model_path "$query_model_name" "$CONFIG_FILE")
response_model_path=$(get_model_path "$response_model_name" "$CONFIG_FILE")


# Execute Python script
python ${MTR_ROOT:-/path/to/mtr}/generate.py \
  --dataset_path "$dataset_path" \
  --query_model_name "$query_model_name" \
  --query_model_path "$query_model_path" \
  --response_model_name $response_model_name \
  --response_model_path $response_model_path \
  --output_dir "$output_dir" \
  --start "$start" \
  --end "$end" \
  --turn "$turn" \
  --cache_dir "$cache_dir" \
  --inference_backend "$inference_backend" \
  --last_turn_dataset $last_turn_dataset