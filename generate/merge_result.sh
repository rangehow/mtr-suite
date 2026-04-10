dataset_dir="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/finance_filtered/Qwen3-235B-A22B-Instruct-2507-Qwen3-235B-A22B-Instruct-2507"
output_dir="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/finance_filtered-final"

# dataset_dir="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/finance_unfiltered/Qwen3-235B-A22B-Instruct-2507-Qwen3-235B-A22B-Instruct-2507"
# output_dir="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/finance_unfiltered-final"

python ${MTR_ROOT:-/path/to/mtr}/generate/merge_result.py --dataset_dir $dataset_dir --output_dir $output_dir
