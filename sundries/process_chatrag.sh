#!/bin/bash

# Define your paths here
CHATRAG_BENCH_BASE_DIR="${BASE_DIR:-/path/to}/dataset/ChatRAG-Bench"
OUTPUT_DATASETS_BASE_DIR="${HOME_DIR:-/path/to}/datasets"
TOPIOCQA_MODIFY_DIR="${BASE_DIR:-/path/to}dataset/ChatRAG-Bench/data/topiocqa/modified_dev.json"
topiocqa_doc_dir="${BASE_DIR:-/path/to}dataset/full_wiki_segments.tsv"
# Ensure the output directory exists (optional, Python script can also create subdirs if needed)
# mkdir -p "${OUTPUT_DATASETS_BASE_DIR}" # Python os.makedirs will handle subdirectories

# Path to your python interpreter (if not in PATH or you want a specific one)
# PYTHON_EXE="/path/to/your/python3"


# Name of your python script
PYTHON_SCRIPT=${MTR_ROOT:-/path/to/mtr}/sundries/process_chatrag.py

echo "Running Python script: ${PYTHON_SCRIPT}"
echo "ChatRAG Bench Path: ${CHATRAG_BENCH_BASE_DIR}"
echo "Output Datasets Path: ${OUTPUT_DATASETS_BASE_DIR}"

# Execute the python script with the arguments
python ${PYTHON_SCRIPT} \
    --chatrag_bench_path "${CHATRAG_BENCH_BASE_DIR}" \
    --output_datasets_path "${OUTPUT_DATASETS_BASE_DIR}"\
    --topiocqa_modify_dir $TOPIOCQA_MODIFY_DIR \
    --topiocqa_doc_dir $topiocqa_doc_dir

echo "Script finished."