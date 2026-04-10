#!/bin/bash
source ${HOME_DIR:-/path/to}/.bashrc
mamba activate sglang
cd ${MTR_ROOT:-/path/to/mtr}/eval

# --- Configuration ---
embedding_output_path='${MTR_ROOT:-/path/to/mtr}/tempfile/embedding'
#
# :::
DELIMITER=":::"

# --- Embedding Model Info (Name${DELIMITER}Path ) ---
declare -a embedding_model_info=(
 # "gte-Qwen2-7B-instruct${DELIMITER}Alibaba-NLP/gte-Qwen2-7B-instruct"
 # "stella_en_400m_v5${DELIMITER}dunzhang/stella_en_400M_v5"
 # "DocChat-Context${DELIMITER}cerebras/Dragon-DocChat-Context-Encoder"
    "bge-large-en-v1.5${DELIMITER}${BASE_DIR:-/path/to}models/bge-large-en-v1.5"
 # "modernbert-base${DELIMITER}Alibaba-NLP/gte-modernbert-base"
 # "chatqa-modernbert-base${DELIMITER}${MTR_ROOT:-/path/to/mtr}/tempfile/model/chatrag_moderbert_base/checkpoint-52"
 # "mtr-modernbert-base${DELIMITER}${MTR_ROOT:-/path/to/mtr}/tempfile/model/mtr_moderbert_base/checkpoint-48"
    "ChatQA-Context${DELIMITER}nvidia/dragon-multiturn-context-encoder"
)

# --- Dataset Info (Name${DELIMITER}Path ) ---
declare -a dataset_info=(
 # "doc2dial${DELIMITER}${HOME_DIR:-/path/to}/datasets/doc2dial_document"
 # "qrecc${DELIMITER}${HOME_DIR:-/path/to}/datasets/qrecc_document"
 # "quac${DELIMITER}${HOME_DIR:-/path/to}/datasets/quac_document"
 # "mtr${DELIMITER}${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/processed_dataset"
 # "topiocqa${DELIMITER}${HOME_DIR:-/path/to}/datasets/topiocqa_document"
    "mtr_finance_filtered${DELIMITER}${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/processed_finance_filtered"
    "mtr_finance_unfiltered${DELIMITER}${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/processed_finance_unfiltered"
)

# --- ---
# 1. dataset_info
declare -a temp_dataset_paths=()
declare -a temp_dataset_names=()

for item in "${dataset_info[@]}"; do
 # : ${string%%delimiter*} ( delimiter )
    current_dataset_name="${item%%${DELIMITER}*}"
 # : ${string#*delimiter} ( delimiter )
    current_dataset_path="${item#*${DELIMITER}}"
    
    temp_dataset_names+=("$current_dataset_name")
    temp_dataset_paths+=("$current_dataset_path")
done

# 2.
dataset_path_str=$(IFS=,; echo "${temp_dataset_paths[*]}")
dataset_name_str=$(IFS=,; echo "${temp_dataset_names[*]}")

# dataset_path_str dataset_name_str
# echo "Debug: dataset_path_str = $dataset_path_str"
# echo "Debug: dataset_name_str = $dataset_name_str"

# --- embedding_model_info (Model) ---
if [ ${#embedding_model_info[@]} -eq 0 ]; then
 echo " embedding Model (embedding_model_info)"
    exit 1
fi
# --- dataset_info () ---
if [ ${#dataset_info[@]} -eq 0 ]; then
 echo " (dataset_info)"
    exit 1
fi


echo "--- Embedding ---"

# --- for embedding Model ---
for model_entry in "${embedding_model_info[@]}"; do
 # Model
    current_model_name="${model_entry%%${DELIMITER}*}"
    current_model_path="${model_entry#*${DELIMITER}}"

 echo "-----------------------------------------"
 echo "--- Model: $current_model_name ---"
 echo " : $current_model_path"
 echo "-----------------------------------------"

 # --- ModelExecute Python script ---
    python document_embedding_st.py \
        --embedding_output_path "$embedding_output_path" \
        --embedding_model_path "$current_model_path" \
        --embedding_model_name "$current_model_name" \
        --dataset_path "$dataset_path_str" \
        --dataset_name "$dataset_name_str"

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