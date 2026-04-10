#!/bin/bash
# Post-processing: merge shards → humanize queries → train/test split
#
# Usage:
#   bash generate/run_postprocess.sh              # full pipeline
#   bash generate/run_postprocess.sh --skip-humanize  # merge + split only (fast)

set -uo pipefail

# ======== Config ========
DATA_DIR="mtr-data-dumps/mtr/Qwen3.5-FP8-Qwen3.5-FP8"
OUTPUT_DIR="mtr-data-dumps/mtr_final"
MODEL_ID="qwen35-fp8"
MAX_CONCURRENT=200
TEST_SIZE=1000

ENDPOINTS=(
    "http://<node1-ip>:8080/v1/chat/completions"
    "http://<node2-ip>:8080/v1/chat/completions"
    # Add more endpoints as needed
)
# =========================

SKIP_HUMANIZE=""
if [[ "${1:-}" == "--skip-humanize" ]]; then
    SKIP_HUMANIZE="--skip_humanize"
    echo "⚡ Skipping humanization (merge + split only)"
fi

# Check endpoints
echo "Checking API endpoints..."
LIVE_ENDPOINTS=()
for ep in "${ENDPOINTS[@]}"; do
    host=$(echo "$ep" | sed 's|http://||;s|/.*||')
    if curl -s --max-time 3 "http://${host}/v1/models" > /dev/null 2>&1; then
        echo "  ✅ $host"
        LIVE_ENDPOINTS+=("$ep")
    else
        echo "  ❌ $host (down, skipping)"
    fi
done

if [ ${#LIVE_ENDPOINTS[@]} -eq 0 ]; then
    echo "ERROR: No live endpoints!"
    exit 1
fi

ENDPOINTS_STR=$(IFS=,; echo "${LIVE_ENDPOINTS[*]}")

echo ""
echo "============================================================"
echo " MTR Post-Processing Pipeline"
echo "============================================================"
echo "  Data:     $DATA_DIR"
echo "  Output:   $OUTPUT_DIR"
echo "  Nodes:    ${#LIVE_ENDPOINTS[@]} live"
echo "  Test:     $TEST_SIZE samples"
echo "  Humanize: $([ -z "$SKIP_HUMANIZE" ] && echo 'YES' || echo 'SKIP')"
echo "============================================================"
echo ""

PYTHONPATH="" python3 generate/merge_and_humanize.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --endpoints "$ENDPOINTS_STR" \
    --model_id "$MODEL_ID" \
    --max_concurrent "$MAX_CONCURRENT" \
    --test_size "$TEST_SIZE" \
    $SKIP_HUMANIZE

echo ""
echo "============================================================"
echo " ✅ Pipeline complete!"
echo "============================================================"
echo "  Train: $OUTPUT_DIR/mtr_train"
echo "  Test:  $OUTPUT_DIR/mtr_test"
echo ""
echo "To inspect:"
echo "  python -c \"import datasets; ds=datasets.load_from_disk('$OUTPUT_DIR/mtr_train'); print(len(ds), ds[0]['messages'][:4])\""
