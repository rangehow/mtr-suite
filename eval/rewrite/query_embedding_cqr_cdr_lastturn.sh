#!/bin/bash

# --- Configuration ---
embedding_output_path='${MTR_ROOT:-/path/to/mtr}/tempfile/embedding'
#
# :::
DELIMITER=":::"

# --- Embedding Model Info (Name${DELIMITER}Path ) ---
declare -a embedding_model_info=(
 # "gte-Qwen2-7B-instruct${DELIMITER}Alibaba-NLP/gte-Qwen2-7B-instruct"
 # "stella_en_400m_v5${DELIMITER}dunzhang/stella_en_400M_v5"
 # "DocChat-Context${DELIMITER}cerebras/Dragon-DocChat-Query-Encoder"
 # "bge-large-en-v1.5${DELIMITER}${BASE_DIR:-/path/to}models/bge-large-en-v1.5"
 # "modernbert-base${DELIMITER}Alibaba-NLP/gte-modernbert-base"
    "chatqa-modernbert-base${DELIMITER}${MTR_ROOT:-/path/to/mtr}/tempfile/model/chatrag_moderbert_base/checkpoint-52"
    "mtr-modernbert-base${DELIMITER}${MTR_ROOT:-/path/to/mtr}/tempfile/model/mtr_moderbert_base/checkpoint-48"
)

# --- Dataset Info (Name${DELIMITER}Path ) ---
declare -a dataset_info=(
    "mtr-rewrite-3B${DELIMITER}${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/mtr-rewrite/Qwen2.5-3B-Instruct"
    "mtr-rewrite-7B${DELIMITER}${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/mtr-rewrite/Qwen2.5-7B-Instruct"
    "mtr-rewrite-14B${DELIMITER}${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/mtr-rewrite/Qwen2.5-14B-Instruct"
    "mtr-rewrite-32B${DELIMITER}${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/mtr-rewrite/Qwen2.5-32B-Instruct"
    "mtr-rewrite-72B${DELIMITER}${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/mtr-rewrite/Qwen2.5-72B-Instruct"
)

# --- ( Python ) ---
# dataset_info
dataset_path_list=""
dataset_name_list=""

for ds_info_str in "${dataset_info[@]}"; do
 # Bash
    ds_name="${ds_info_str%%${DELIMITER}*}"
    ds_path="${ds_info_str#*${DELIMITER}}"

    if [ -z "$dataset_name_list" ]; then
        dataset_name_list="$ds_name"
        dataset_path_list="$ds_path"
    else
        dataset_name_list="${dataset_name_list},$ds_name"
        dataset_path_list="${dataset_path_list},$ds_path"
    fi
done

echo "--- Embedding ---"

# --- for embedding Model ---
# ( embedding_model_info)

for model_info_str in "${embedding_model_info[@]}"; do
 # Bash
    current_model_name="${model_info_str%%${DELIMITER}*}"
    current_model_path="${model_info_str#*${DELIMITER}}"

 echo "-----------------------------------------"
 echo "--- Model: $current_model_name ---"
 echo " : $current_model_path"
 echo " : $dataset_name_list"
 echo " Dataset path: $dataset_path_list"
 echo "-----------------------------------------"

 # --- ModelExecute Python script ---
    python query_embedding_cqr_cdr_lastturn.py \
        --embedding_output_path "$embedding_output_path" \
        --embedding_model_path "$current_model_path" \
        --embedding_model_name "$current_model_name" \
        --dataset_path "$dataset_path_list" \
        --dataset_name "$dataset_name_list"

 # Python ()
    if [ $? -ne 0 ]; then
 echo "Model $current_model_name Python Script execution failed!"
 # Model
 # exit 1 #
    fi
 echo # Model
done

echo "--- Model ---"
echo "--- ---"