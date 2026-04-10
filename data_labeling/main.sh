#!/bin/bash

# Exit on error
set -e

# --- Configuration ---
CSV_FILE="scores_qrecc.csv"

HF_DATASET_PATHS=(
    "${MTR_ROOT:-/path/to/mtr}/tempfile/chatrag/Athene-V2-Chat_qrecc"
    "${MTR_ROOT:-/path/to/mtr}/tempfile/chatrag/Gemma-3-27b-it_qrecc"
    "${MTR_ROOT:-/path/to/mtr}/tempfile/chatrag/GLM-4-32B-0414_qrecc"
    "${MTR_ROOT:-/path/to/mtr}/tempfile/chatrag/Llama-4-Scout-17B-16E-Instruct_qrecc"
    "${MTR_ROOT:-/path/to/mtr}/tempfile/chatrag/Mistral-Large-Instruct-2411_qrecc"
    "${MTR_ROOT:-/path/to/mtr}/tempfile/chatrag/Qwen2.5-72B-Instruct_qrecc"
    "${MTR_ROOT:-/path/to/mtr}/tempfile/chatrag/Command-a_qrecc"
)

PYTHON_SCRIPT="main.py" # Using Python script name
PYTHON_EXECUTABLE="python3"

# Optional: HuggingFace DatasetDict split preference (passed to Python script)
# If your HF datasets are DatasetDicts and you want to consistently use e.g. 'train' split
# you can pass via --hf_preferred_splits arg. Python script defaults to ['train', 'validation', 'test']
HF_SPLIT_ARGS="--hf_preferred_splits train" # Example: only use 'train'，or "train validation test" try in order
# HF_SPLIT_ARGS="" # Leave empty if defaults are fine or dataset is not DatasetDict

# --- End configuration ---

# Check if Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
 echo "Error: Python script '${PYTHON_SCRIPT}' not found!"
    exit 1
fi

# Check if CSV file exists
if [ ! -f "${CSV_FILE}" ]; then
 echo "Error: CSV file '${CSV_FILE}' not found!"
    exit 1
fi

# Check if HF_DATASET_PATHS is empty
if [ ${#HF_DATASET_PATHS[@]} -eq 0 ]; then
 echo "Error: HF dataset paths list (HF_DATASET_PATHS) is empty!"
    exit 1
fi

# Verify all HF dataset paths exist as directories (Python does more detailed checks)
for hf_path in "${HF_DATASET_PATHS[@]}"; do
    if [ ! -d "${hf_path}" ]; then
 echo "Warning: HF dataset directory '${hf_path}' not found! Script will try to continue but Python may fail."
 # If strict checking desired, exit here: exit 1
    fi
done

echo "========================================"
echo "Starting aggregation evaluation..."
echo "CSV file: ${CSV_FILE}"
echo "Hugging Face : ${#HF_DATASET_PATHS[@]}"
echo "Python : ${PYTHON_SCRIPT}"
if [ -n "$HF_SPLIT_ARGS" ]; then
 echo "HF Split : ${HF_SPLIT_ARGS}"
fi
echo "----------------------------------------"

# Execute Python script CSV file HF Dataset path
# HF_DATASET_PATHS
# ${HF_SPLIT_ARGS} containing
# shellcheck disable=SC2086 # $HF_SPLIT_ARGS
${PYTHON_EXECUTABLE} "${PYTHON_SCRIPT}" "${CSV_FILE}" "${HF_DATASET_PATHS[@]}" ${HF_SPLIT_ARGS}

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
 echo "----------------------------------------"
 echo "Python : $EXIT_CODE"
 echo "========================================"
    exit $EXIT_CODE
fi

echo "----------------------------------------"
echo "Python Script execution complete."
echo "========================================"