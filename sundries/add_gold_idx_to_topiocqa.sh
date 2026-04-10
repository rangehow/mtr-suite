#!/bin/bash

#
TSV_FILE="${BASE_DIR:-/path/to}dataset/full_wiki_segments.tsv"
INPUT_JSON="${BASE_DIR:-/path/to}dataset/ChatRAG-Bench/data/topiocqa/dev.json"
OUTPUT_JSON="${BASE_DIR:-/path/to}dataset/ChatRAG-Bench/data/topiocqa/modified_dev.json"

# Check if Python script exists
PYTHON_SCRIPT="${MTR_ROOT:-/path/to/mtr}/sundries/add_gold_idx_to_topiocqa.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
 echo "Error: Python script '$PYTHON_SCRIPT' not found"
 echo " shell or"
    exit 1
fi

# Execute Python script
echo "Execute Python script..."
python "$PYTHON_SCRIPT" "$TSV_FILE" "$INPUT_JSON" "$OUTPUT_JSON"

# Python
if [ $? -eq 0 ]; then
 echo "Python "
else
 echo "Python "
fi