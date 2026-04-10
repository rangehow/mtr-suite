#!/bin/bash

# ==============================
# HOPE
# Model1 , Model2
# ==============================

# --- Configuration ---
HOPE_TEMPLATE="template.hope" # Path to your HOPE template file
MAIN_SCRIPT_PATH="main.sh"    # Path to your main.sh script
TURN=1
START=0
END=499
# ----------------

# --- Model ( model_name_1 ) ---
declare -a model_list=(
  "Mistral-Large-Instruct-2411"
  "Qwen2.5-72B-Instruct"
  "Gemma-3-27b-it"
  "Llama-4-Scout-17B-16E-Instruct"
  "Athene-V2-Chat"
  "Command-a"
  "GLM-4-32B-0414"
)
# -----------------------------------------

# --- Model () ---
FIXED_MODEL_2="Qwen3-235B-A22B-Instruct-2507"

# ./run_hope_tasks.sh "Gemma-3-27b-it"
if [ $# -ge 1 ]; then
    FIXED_MODEL_2="$1"
 echo ">>> model_name_2: ${FIXED_MODEL_2}"
else
 echo ">>> model_name_2: ${FIXED_MODEL_2}"
fi
# -------------------------------------------------

#
if [ ! -f "$HOPE_TEMPLATE" ]; then
 echo "Error: HOPE template file '$HOPE_TEMPLATE' not found"
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT_PATH" ]; then
 echo "Error: main script '$MAIN_SCRIPT_PATH' not found"
    exit 1
fi

#
declare -a temp_hope_files=()
declare -a bg_pids=()

echo "==============================================="
echo "Model HOPE :"
printf -- "- %s\n" "${model_list[@]}"
echo "Model: ${FIXED_MODEL_2}"
echo "Turn=${TURN}, Start=${START}, End=${END}"
echo "==============================================="

# : model_name_1 , model_name_2
for model_name_1 in "${model_list[@]}"; do
  model_name_2="${FIXED_MODEL_2}"

  worker_command="bash ${MAIN_SCRIPT_PATH} \"${model_name_1}\" ${TURN} ${START} ${END} \"${model_name_2}\" ${TARGET}"

 #
 safe_model_name_1=$(echo "${model_name_1}" | tr -cd '[:alnum:]_-')
  [ -z "$safe_model_name_1" ] && safe_model_name_1="model1_$(date +%s%N)_${RANDOM}"

 safe_model_name_2=$(echo "${model_name_2}" | tr -cd '[:alnum:]_-')
  [ -z "$safe_model_name_2" ] && safe_model_name_2="model2_$(date +%s%N)_${RANDOM}"

  temp_hope_file="choice_${safe_model_name_1}_vs_${safe_model_name_2}.hope"

 # hope
  sed "s#^worker.script = .*#worker.script = ${worker_command}#" "${HOPE_TEMPLATE}" > "${temp_hope_file}"
  if [ $? -ne 0 ]; then
 echo "❌ ${temp_hope_file} "
      rm -f "${temp_hope_file}"
      continue
  fi

 echo "✅ : ${temp_hope_file}"
 echo ": ${worker_command}"

 #
  hope run "${temp_hope_file}" &
  if [ $? -eq 0 ]; then
      bg_pids+=($!)
      temp_hope_files+=("${temp_hope_file}")
 echo "🚀 (${model_name_1} vs ${model_name_2}) PID=$!"
  else
 echo "⚠️ : ${model_name_1} vs ${model_name_2}"
      rm -f "${temp_hope_file}"
  fi
 echo "-----------------------------------------------"

done

echo "Waiting for all..."
wait
echo "✅ HOPE "

#
echo "..."
for tmp_file in "${temp_hope_files[@]}"; do
  [ -f "$tmp_file" ] && rm "$tmp_file"
done
echo "🧹 "

echo "🎯 Script execution complete."
