#!/bin/bash
cd ${MTR_ROOT:-/path/to/mtr}/generate
# --- Configuration ---
HOPE_TEMPLATE="template.hope" # Path to your HOPE template file
MAIN_SCRIPT_PATH="main.sh"    # Path to your main.sh script
TURN=4                        # Fixed turn parameter

# Global task range
GLOBAL_START=0
GLOBAL_END=6047 # Example: total range from 0 to 999999 (1M)

# Length per task
TASK_LENGTH=500

# Total number of tasks to submit
NUM_TASKS=1 # e.g.: submit 5 tasks of length 10000

# dataset_path="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/cluster_dataset/finance_filtered"
# output_dir="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/finance_filtered"
# cache_dir="${MTR_ROOT:-/path/to/mtr}/tempfile/finance_filtered_cache"

dataset_path="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/cluster_dataset/finance_unfiltered"
output_dir="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/finance_unfiltered"
cache_dir="${MTR_ROOT:-/path/to/mtr}/tempfile/finance_unfiltered_cache"

# --- Fixed model names ---
# query_model_name="Mistral-Large-Instruct-2411"
# response_model_name="Mistral-Large-Instruct-2411"

query_model_name="Qwen3-235B-A22B-Instruct-2507"
response_model_name="Qwen3-235B-A22B-Instruct-2507"  

# ------------------

# Check if template file exists
if [ ! -f "$HOPE_TEMPLATE" ]; then
 echo "Error: HOPE template file '$HOPE_TEMPLATE' not found"
    exit 1
fi

# Check if main.sh exists
if [ ! -f "$MAIN_SCRIPT_PATH" ]; then
 echo "Error: main script '$MAIN_SCRIPT_PATH' not found"
    exit 1
fi

# Calculate total range size
TOTAL_RANGE_SIZE=$((GLOBAL_END - GLOBAL_START + 1))

# Task length
if [ "$TASK_LENGTH" -le 0 ]; then
 echo "Task length ($TASK_LENGTH) must be greater than 0"
    exit 1
fi
if [ "$TASK_LENGTH" -gt "$TOTAL_RANGE_SIZE" ]; then
 echo "Task length ($TASK_LENGTH) exceeds total range size ($TOTAL_RANGE_SIZE)"
    exit 1
fi

# Calculate total length of all tasks
TOTAL_TASKS_LENGTH=$((TASK_LENGTH * NUM_TASKS))

# Calculate available space ()
AVAILABLE_SPACE=$((TOTAL_RANGE_SIZE - TASK_LENGTH)) # Available start point range size

# If number of tasks exceeds available start points
if [ "$NUM_TASKS" -gt $((AVAILABLE_SPACE + 1)) ]; then
 echo "Number of tasks ($NUM_TASKS) $TASK_LENGTH ($((AVAILABLE_SPACE + 1)))"
 echo " GLOBAL_START, GLOBAL_END, TASK_LENGTH NUM_TASKS "
fi


if [ "$NUM_TASKS" -gt 1 ]; then
   INTERVAL_SIZE=$(( (GLOBAL_END - GLOBAL_START - TASK_LENGTH + 1) / (NUM_TASKS - 1) ))
else
   INTERVAL_SIZE=0 # For a single task, interval is 0
fi
if [ "$NUM_TASKS" -gt 1 ] && [ "$INTERVAL_SIZE" -lt 1 ]; then
    INTERVAL_SIZE=1 # Minimum interval is 1, tasks may overlap
 echo "11"
fi

# Store generated temp hope file names
declare -a temp_hope_files=()
# Store background task PIDs
declare -a bg_pids=()
# CounterNumber of tasks
submitted_task_count=0
# CounterNumber of tasks
skipped_task_count=0

echo "Model '${query_model_name}' vs '${response_model_name}' HOPE (Turn=${TURN}):"
echo ": ${GLOBAL_START}-${GLOBAL_END}, Task length: ${TASK_LENGTH}, Number of tasks: ${NUM_TASKS}"
echo " (): ${INTERVAL_SIZE}"
echo ""
echo "---"

#
for i in $(seq 0 $((NUM_TASKS - 1))); do
 #
    current_start=$((GLOBAL_START + i * INTERVAL_SIZE))

 # (containing)
    current_end=$((current_start + TASK_LENGTH - 1))

 # Safety check: ensure end does not exceed global end
    if [ ${current_end} -gt ${GLOBAL_END} ]; then
 echo " ${i} ${current_end} ${GLOBAL_END} ${GLOBAL_END}"
         current_end=${GLOBAL_END}
         current_length=$((current_end - current_start + 1))
         if [ "$current_length" -le 0 ]; then
 echo " ${i} ${current_start} ${current_end}"
             skipped_task_count=$((skipped_task_count + 1))
             continue
         fi
         if [ "$current_length" -ne "$TASK_LENGTH" ]; then
 echo " ${i} ${current_length}"
         fi
    fi
    if [ ${current_start} -gt ${current_end} ]; then
 echo " ${i} ${current_start} ${current_end}"
        skipped_task_count=$((skipped_task_count + 1))
        continue
    fi


 # --- ---
    target_dir="${output_dir}/${query_model_name}-${response_model_name}/${TURN}/${current_start}-${current_end}"
 echo " ${i} ( ${current_start}-${current_end}), : ${target_dir} ..."

 # (&&) (-n "$(find...)")
    if [ -d "$target_dir" ] && [ -n "$(find "$target_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
 echo " ${i} ( ${current_start}-${current_end}): '${target_dir}' "
        skipped_task_count=$((skipped_task_count + 1))
        continue # Skip to next iteration
    else
 #
 echo "Conditions met, preparing to submit task ${i} ( ${current_start}-${current_end})..."

 # Construct worker.script command
        worker_command="bash ${MAIN_SCRIPT_PATH} \"${query_model_name}\" ${TURN} ${current_start} ${current_end} \"${response_model_name}\" ${dataset_path} ${output_dir} ${cache_dir}"

 # Create safe filename parts for model names
 safe_model1=$(echo "${query_model_name}" | tr -cd '[:alnum:]_-')
        [ -z "$safe_model1" ] && safe_model1="model1_$(date +%s%N)_${RANDOM}"
 safe_model2=$(echo "${response_model_name}" | tr -cd '[:alnum:]_-')
        [ -z "$safe_model2" ] && safe_model2="model2_$(date +%s%N)_${RANDOM}"

 # Construct temp hope file name
        temp_hope_file="work_${safe_model1}_${safe_model2}_task${i}_${current_start}_${current_end}.hope"

 # Use sed to generate temp .hope
        sed "s#^worker.script = .*#worker.script = ${worker_command}#" "${HOPE_TEMPLATE}" > "${temp_hope_file}"

        if [ $? -ne 0 ]; then
 echo " ${i} HOPE '${temp_hope_file}' "
            skipped_task_count=$((skipped_task_count + 1)) # Counted as skipped
            continue
        fi

 echo " ${i} HOPE : ${temp_hope_file}"
 echo ": ${worker_command}"

 # Submit HOPE task to background
 echo " ${i} for range ${current_start}-${current_end}..."
        hope run "${temp_hope_file}" &

        if [ $? -eq 0 ]; then
            bg_pids+=($!)
            temp_hope_files+=("${temp_hope_file}")
            submitted_task_count=$((submitted_task_count + 1))
 echo " ${i} for range ${current_start}-${current_end} (PID: $!)"
        else
 echo " ${i} "
            skipped_task_count=$((skipped_task_count + 1)) # Submit failure also counted as skipped
 #
            rm -f "${temp_hope_file}"
        fi
    fi
 # --- ---

done

echo "---"
echo "Loop completeTotal planned ${NUM_TASKS} tasks"
echo "Actually submitted ${submitted_task_count} tasks"
echo "Skipped ${skipped_task_count} tasks ()"

if [ ${#bg_pids[@]} -gt 0 ]; then
 echo "Waiting for all ${#bg_pids[@]} background tasks to complete..."
    wait
    wait_exit_status=$?
 echo "All background HOPE processes finished"
 echo " 'hope run' HOPE "

    if [ $wait_exit_status -ne 0 ]; then
 echo " HOPE (wait : $wait_exit_status)"
    fi

 echo "Cleaning up temp HOPE files (${#temp_hope_files[@]} )..."
    for tmp_file in "${temp_hope_files[@]}"; do
      if [ -f "$tmp_file" ]; then
 # echo ": ${tmp_file}" #
        rm "$tmp_file"
      else
 echo ": ${tmp_file} not found"
      fi
    done
 echo "Temp file cleanup complete"
else
 echo "No tasks were submitted for background execution"
fi

echo "Script execution complete."