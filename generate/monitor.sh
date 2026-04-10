#!/bin/bash
#
# Monitor MTR generation progress.
#
# Usage:
#   bash generate/monitor.sh                     # one-shot summary
#   bash generate/monitor.sh --watch             # refresh every 10s
#   bash generate/monitor.sh --watch --interval 5  # refresh every 5s
#
set -euo pipefail

WATCH=false
INTERVAL=10
while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch|-w)  WATCH=true; shift ;;
        --interval)  INTERVAL="$2"; shift 2 ;;
        *)           shift ;;
    esac
done

# ============================================================
# Configuration (match run_api.sh)
# ============================================================
cd "${MTR_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"

MODEL_DISPLAY_NAME="Qwen3.5-FP8"
MODEL_TAG="${MODEL_DISPLAY_NAME}-${MODEL_DISPLAY_NAME}"
OUTPUT_DIR="$(pwd)/mtr-data-dumps/mtr/${MODEL_TAG}"
CACHE_DIR="$(pwd)/tempfile/cache"
TOTAL_TURNS=12
GLOBAL_END=10000
TASK_LENGTH=2500

ENDPOINTS=(
    "http://<node1-ip>:8080/v1/chat/completions"
    "http://<node2-ip>:8080/v1/chat/completions"
    # Add more endpoints as needed
)

# ============================================================

show_progress() {
    clear 2>/dev/null || true
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              MTR Generation Progress Monitor                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    printf "  %-12s %s\n" "Time:" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf "  %-12s %s\n" "Model:" "$MODEL_DISPLAY_NAME"
    printf "  %-12s %s entries × %s turns\n" "Target:" "$GLOBAL_END" "$TOTAL_TURNS"
    echo ""

    # ---- Endpoint health ----
    echo "  ┌─ Endpoints ────────────────────────────────────────────┐"
    LIVE=0
    for url in "${ENDPOINTS[@]}"; do
        host=$(echo "$url" | sed 's|http://||' | sed 's|/.*||')
        if curl -s -m 2 "http://${host}/v1/models" > /dev/null 2>&1; then
            printf "  │  ✅ %-50s │\n" "$host"
            LIVE=$((LIVE + 1))
        else
            printf "  │  ❌ %-50s │\n" "$host (down)"
        fi
    done
    echo "  └────────────────────────────────────────────────────────┘"
    echo ""

    # ---- Per-turn progress ----
    # Calculate expected shard count from config (must match run_api.sh logic)
    EXPECTED_SHARDS=$(( (GLOBAL_END + TASK_LENGTH - 1) / TASK_LENGTH ))

    echo "  ┌─ Turn Progress ────────────────────────────────────────┐"
    TOTAL_DONE=0
    for turn in $(seq 1 "$TOTAL_TURNS"); do
        turn_dir="${OUTPUT_DIR}/${turn}"

        # Count completed shards (dirs with dataset_info.json = finished)
        done_shards=0
        if [ -d "$turn_dir" ]; then
            done_shards=$(find "$turn_dir" -maxdepth 2 -name "dataset_info.json" 2>/dev/null | wc -l)
        fi

        # Determine status
        if [ "$done_shards" -eq 0 ]; then
            status="⬜ pending"
        elif [ "$done_shards" -ge "$EXPECTED_SHARDS" ]; then
            status="✅ done   "
            done_shards=$EXPECTED_SHARDS  # clamp display
            TOTAL_DONE=$((TOTAL_DONE + 1))
        else
            status="🔄 running"
        fi

        # Check cache for query-phase progress
        cache_status=""
        cache_turn_dir="${CACHE_DIR}/${MODEL_DISPLAY_NAME}/${turn}"
        if [ -d "$cache_turn_dir" ]; then
            cache_count=$(find "$cache_turn_dir" -maxdepth 2 -name "dataset_info.json" 2>/dev/null | wc -l)
            if [ "$cache_count" -gt 0 ] && [ "$status" != "✅ done   " ]; then
                cache_status=" (query cached)"
            fi
        fi

        # Build progress bar
        bar_len=20
        if [ "$EXPECTED_SHARDS" -gt 0 ]; then
            filled=$((done_shards * bar_len / EXPECTED_SHARDS))
        else
            filled=0
        fi
        [ "$filled" -gt "$bar_len" ] && filled=$bar_len
        empty=$((bar_len - filled))
        bar=$(printf '█%.0s' $(seq 1 $filled 2>/dev/null) || true)
        bar+=$(printf '░%.0s' $(seq 1 $empty 2>/dev/null) || true)
        printf "  │  Turn %2d/%-2d  %s  [%s] %d/%d shards%s │\n" \
            "$turn" "$TOTAL_TURNS" "$status" "$bar" "$done_shards" "$EXPECTED_SHARDS" "$cache_status"
    done
    echo "  └────────────────────────────────────────────────────────┘"
    echo ""

    # ---- Overall summary ----
    pct=$((TOTAL_DONE * 100 / TOTAL_TURNS))
    echo "  Overall: ${TOTAL_DONE}/${TOTAL_TURNS} turns complete (${pct}%)"
    echo ""

    # ---- Active processes ----
    procs=$(pgrep -f "main_api.py" 2>/dev/null | wc -l)
    echo "  Active processes: ${procs}"
    echo ""

    # ---- Latest log lines ----
    latest_log=$(ls -t "${CACHE_DIR}"/task_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "  ┌─ Latest Log ($(basename "$latest_log")) ──────────────────┐"
        tail -3 "$latest_log" 2>/dev/null | while IFS= read -r line; do
            printf "  │  %.56s │\n" "$line"
        done
        echo "  └────────────────────────────────────────────────────────┘"
    fi
}

if [ "$WATCH" = true ]; then
    while true; do
        show_progress
        echo ""
        echo "  Refreshing every ${INTERVAL}s... (Ctrl+C to stop)"
        sleep "$INTERVAL"
    done
else
    show_progress
fi
