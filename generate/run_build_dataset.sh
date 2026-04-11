#!/bin/bash
# Build the final MTR dataset from generated data.
# Usage: bash generate/run_build_dataset.sh

ENDPOINTS="http://<node1-ip>:8080/v1/chat/completions,http://<node2-ip>:8080/v1/chat/completions,http://<node3-ip>:8080/v1/chat/completions,http://<node4-ip>:8080/v1/chat/completions"

PYTHONPATH="" python generate/build_dataset.py \
    --data_dir mtr-data-dumps/mtr/Qwen3.5-FP8-Qwen3.5-FP8 \
    --output_dir mtr-data-dumps/mtr_dataset \
    --endpoints "$ENDPOINTS" \
    --model_id qwen35-fp8 \
    --max_concurrent 200 \
    --total_turns 12 \
    --test_size 1000 \
    --switch_ratio 0.5 \
    --seed 42 \
    --cache_dir tempfile/cache/build_dataset_cache
