#!/bin/bash
source ${HOME_DIR:-/path/to}/.bashrc
mamba activate faiss
cd ${MTR_ROOT:-/path/to/mtr}/eval
# --- Configuration ---
embedding_output_path='${MTR_ROOT:-/path/to/mtr}/tempfile/embedding'
index_output_path='${MTR_ROOT:-/path/to/mtr}/tempfile/index'
# --- Embedding Model Arrays ---
# declare -a

declare -a embedding_model_name=(
 # "gte-Qwen2-7B-instruct"
 # "stella_en_400m_v5"
 # "DocChat-Context"
    "bge-large-en-v1.5"
 # "modernbert-base"
 # "chatqa-modernbert-base"
 # "mtr-modernbert-base"
    "ChatQA-Context"
)

# --- Dataset Arrays ---
declare -a dataset_name=(
 # "mtr"
 # "doc2dial"
 # "qrecc"
 # "quac"
 # "topiocqa"
    "mtr_finance_filtered"
    "mtr_finance_unfiltered"
)

# --- ( Python ) ---

dataset_name_str=$(IFS=,; echo "${dataset_name[*]}")


echo "--- Embedding ---"

# --- for embedding Model ---
num_models=${#embedding_model_name[@]}
for (( i=0; i<num_models; i++ )); do
 # found
    current_model_name="${embedding_model_name[i]}"

 echo "-----------------------------------------"
 echo "--- Model: $current_model_name ---"
 echo "-----------------------------------------"

 # --- ModelExecute Python script ---
    python ${MTR_ROOT:-/path/to/mtr}/eval/document_index.py \
        --embedding_output_path "$embedding_output_path" \
        --embedding_model_name "$current_model_name" \
        --dataset_name "$dataset_name_str" \
        --index_output_path $index_output_path

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