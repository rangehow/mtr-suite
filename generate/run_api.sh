#!/bin/bash
#
# MTR generation using remote sglang API endpoints (5× Qwen3.5-FP8 nodes).
#
# Generates multi-turn retrieval dialogues from clustered Wikipedia documents.
# Each task processes a shard sequentially; within each task, requests are sent
# with high async concurrency across all endpoint nodes.
#
# Usage:
#   bash generate/run_api.sh              # full run (123K clusters, 4 turns)
#   bash generate/run_api.sh --dry-run    # show plan without executing
#
# Prerequisites:
#   1. Clustered dataset at $dataset_path
#   2. sglang servers running at the endpoints below
#   3. pip install aiohttp loguru tqdm datasets
#

set -uo pipefail
# Note: -e removed intentionally — background jobs + wait need manual error handling
cd "${MTR_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"

# Avoid swebench PYTHONPATH pollution
export PYTHONPATH=""

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ============================================================
# Configuration
# ============================================================

# Remote sglang endpoints (Qwen3.5-FP8)
# Add your sglang/vllm endpoint URLs below (comma-separated)
ENDPOINTS="http://<node1-ip>:8080/v1/chat/completions"
ENDPOINTS+=",http://<node2-ip>:8080/v1/chat/completions"
# ENDPOINTS+=",http://<node3-ip>:8080/v1/chat/completions"

# Model settings
MODEL_ID="qwen35-fp8"
MODEL_DISPLAY_NAME="Qwen3.5-FP8"

# Dataset path — your 123K pre-clustered Wikipedia dataset
dataset_path="mtr-data-dumps/cluster_dataset"
output_dir="$(pwd)/mtr-data-dumps/mtr"
cache_dir="$(pwd)/tempfile/cache"

# Generation parameters
TURN=12                   # Number of dialogue turns per cluster
GLOBAL_START=0            # Start index
GLOBAL_END=10000          # 10K clusters → 10K multi-turn conversations
TASK_LENGTH=2500          # Samples per parallel task shard (4 shards total)
MAX_CONCURRENT=200        # Async in-flight requests per task (4 nodes × ~50 each)
ENABLE_THINKING="false"   # Disable reasoning (10× faster, sufficient for synthesis)
MAX_NEW_TOKENS=2048       # Sufficient for queries (~50 tok) and responses (~500 tok)
TEMPERATURE=0.8
TOP_P=0.95

# Parallelism: how many task shards to run simultaneously
# With 4 nodes and 200 concurrent per task, 2 parallel tasks saturate the cluster
MAX_PARALLEL_TASKS=2

# ============================================================
# Health check
# ============================================================

echo "============================================================"
echo " MTR API Generation Pipeline"
echo "============================================================"

LIVE_NODES=0
for url in $(echo "$ENDPOINTS" | tr ',' '\n'); do
    host=$(echo "$url" | sed 's|http://||' | sed 's|/.*||')
    if curl -s -m 3 "http://${host}/v1/models" > /dev/null 2>&1; then
        echo "  ✅ ${host}"
        LIVE_NODES=$((LIVE_NODES + 1))
    else
        echo "  ❌ ${host} (unreachable)"
    fi
done
echo ""

if [ "$LIVE_NODES" -eq 0 ]; then
    echo "ERROR: No live endpoints. Aborting."
    exit 1
fi

# ============================================================
# Plan
# ============================================================

NUM_TASKS=$(( (GLOBAL_END - GLOBAL_START + TASK_LENGTH - 1) / TASK_LENGTH ))

echo "Model:       ${MODEL_DISPLAY_NAME} (${MODEL_ID})"
echo "Live nodes:  ${LIVE_NODES}"
echo "Dataset:     ${dataset_path}"
echo "Range:       ${GLOBAL_START} → ${GLOBAL_END} (${GLOBAL_END} clusters)"
echo "Turns:       ${TURN}"
echo "Shards:      ${NUM_TASKS} × ${TASK_LENGTH} samples"
echo "Parallel:    up to ${MAX_PARALLEL_TASKS} shards at a time"
echo "Concurrent:  ${MAX_CONCURRENT} async requests per shard"
echo "Thinking:    ${ENABLE_THINKING}"
echo "Output:      ${output_dir}"
echo "============================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would launch ${NUM_TASKS} tasks. Exiting."
    exit 0
fi

# Create output dirs
mkdir -p "$output_dir" "$cache_dir"

# ============================================================
# Launch tasks with controlled parallelism
# ============================================================

declare -a PIDS=()
SUBMITTED=0
SKIPPED=0
RUNNING=0

wait_for_slot() {
    while [ "$RUNNING" -ge "$MAX_PARALLEL_TASKS" ]; do
        # Wait for any child to finish
        for idx in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$idx]}" 2>/dev/null; then
                wait "${PIDS[$idx]}" || true
                unset 'PIDS[$idx]'
                RUNNING=$((RUNNING - 1))
            fi
        done
        # Compact array
        PIDS=("${PIDS[@]}")
        [ "$RUNNING" -ge "$MAX_PARALLEL_TASKS" ] && sleep 5
    done
}

for i in $(seq 0 $((NUM_TASKS - 1))); do
    task_start=$((GLOBAL_START + i * TASK_LENGTH))
    task_end=$((task_start + TASK_LENGTH))

    # Clamp to global end
    [ "$task_end" -gt "$GLOBAL_END" ] && task_end=$GLOBAL_END

    # Skip empty range
    [ "$task_start" -ge "$task_end" ] && { SKIPPED=$((SKIPPED + 1)); continue; }

    # Skip if already completed (last turn output exists)
    target_dir="${output_dir}/${MODEL_DISPLAY_NAME}-${MODEL_DISPLAY_NAME}/${TURN}/${task_start}-${task_end}"
    if [ -d "$target_dir" ] && [ -f "${target_dir}/dataset_info.json" ]; then
        echo "[Shard ${i}] SKIP: ${task_start}-${task_end} (already done)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Wait for a parallel slot
    wait_for_slot

    echo "[Shard ${i}] Launching: ${task_start} → ${task_end}"
    mkdir -p "${cache_dir}"

    python generate/main_api.py \
        --dataset_path "$dataset_path" \
        --query_endpoints "$ENDPOINTS" \
        --model_id "$MODEL_ID" \
        --model_display_name "$MODEL_DISPLAY_NAME" \
        --output_dir "$output_dir" \
        --start "$task_start" \
        --end "$task_end" \
        --turn "$TURN" \
        --cache_dir "$cache_dir" \
        --max_concurrent "$MAX_CONCURRENT" \
        --enable_thinking "$ENABLE_THINKING" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        > "${cache_dir}/task_${task_start}_${task_end}.log" 2>&1 &

    PIDS+=($!)
    SUBMITTED=$((SUBMITTED + 1))
    RUNNING=$((RUNNING + 1))
done

echo ""
echo "Submitted: ${SUBMITTED} tasks, Skipped: ${SKIPPED}"

# Wait for remaining tasks
if [ ${#PIDS[@]} -gt 0 ]; then
    echo "Waiting for final ${#PIDS[@]} tasks..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            FAILED=$((FAILED + 1))
        fi
    done
    echo ""
    echo "========================================="
    echo " COMPLETE"
    echo " Total: ${SUBMITTED}, Failed: ${FAILED}"
    echo " Output: ${output_dir}"
    echo "========================================="
else
    echo "No tasks were launched (all done or skipped)."
fi
