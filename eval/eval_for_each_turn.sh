#!/bin/bash
source ${HOME_DIR:-/path/to}/.bashrc
mamba activate sglang
cd ${MTR_ROOT:-/path/to/mtr}/eval

# --- Configuration ---
embedding_output_path='${MTR_ROOT:-/path/to/mtr}/tempfile/embedding'
index_output_path='${MTR_ROOT:-/path/to/mtr}/tempfile/index'
results_save_dir='${MTR_ROOT:-/path/to/mtr}/tempfile/eval_results'
declare -a embedding_model_name=(
    "ChatQA-Context"
    "DocChat-Context"
    "chatqa-modernbert-base"
    "mtr-modernbert-base"
    "gte-Qwen2-7B-instruct"
    "stella_en_400m_v5"
    "modernbert-base"
    "bge-large-en-v1.5"
)

# --- Dataset "Tuples" ---
# : "domain_map_path;query_dataset_path;dataset_name"
# (;) containing
declare -a dataset_configs=(
 # "${HOME_DIR:-/path/to}/datasets/doc2dial_document/domain_map.json;${HOME_DIR:-/path/to}/datasets/doc2dial;doc2dial"
 # "${HOME_DIR:-/path/to}/datasets/qrecc_document/domain_map.json;${HOME_DIR:-/path/to}/datasets/qrecc;qrecc"
 # "${HOME_DIR:-/path/to}/datasets/quac_document/domain_map.json;${HOME_DIR:-/path/to}/datasets/quac;quac"
 # "full;${HOME_DIR:-/path/to}/datasets/topiocqa;topiocqa"
    "full;${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/mtr_test;mtr"

)

# --- "dataset_configs" ---
declare -a temp_domain_map_paths=()
declare -a temp_query_dataset_paths=()
declare -a temp_dataset_names=()

for config_str in "${dataset_configs[@]}"; do
 # #
 # Bash #
    if [[ -z "$config_str" ]]; then
        continue
    fi

 # IFS
    OLD_IFS="$IFS"
    IFS=';'
 # read
 # -r
    read -r current_domain_map current_query_dataset current_dataset_name <<< "$config_str"
    IFS="$OLD_IFS" # Restore IFS

    temp_domain_map_paths+=("$current_domain_map")
    temp_query_dataset_paths+=("$current_query_dataset")
    temp_dataset_names+=("$current_dataset_name")
done

# --- ( Python ) ---
domain_map_path_str=$(IFS=,; echo "${temp_domain_map_paths[*]}")
dataset_name_str=$(IFS=,; echo "${temp_dataset_names[*]}")
query_dataset_path_str=$(IFS=,; echo "${temp_query_dataset_paths[*]}")


echo "--- ---"
echo "Domain Map Paths: $domain_map_path_str" #
echo "Dataset Names: $dataset_name_str" #
echo "Query Dataset Paths: $query_dataset_path_str" #

# --- for embedding Model ---
num_models=${#embedding_model_name[@]}
for (( i=0; i<num_models; i++ )); do
 # found
    current_model_name="${embedding_model_name[i]}"

 echo "-----------------------------------------"
 echo "--- Model: $current_model_name ---"
 echo "-----------------------------------------"

 # --- ModelExecute Python script ---
    python ${MTR_ROOT:-/path/to/mtr}/eval/eval_for_each_turn.py \
        --embedding_output_path "$embedding_output_path" \
        --embedding_model_name "$current_model_name" \
        --domain_map_path "$domain_map_path_str" \
        --dataset_name "$dataset_name_str" \
        --index_output_path "$index_output_path" \
        --query_dataset_path "$query_dataset_path_str"\
        --results_save_dir $results_save_dir

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