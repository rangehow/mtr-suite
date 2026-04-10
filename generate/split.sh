dataset_path="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/mtr-bench-phrased"
output_dir="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps"

python ${MTR_ROOT:-/path/to/mtr}/generate/split.py --dataset_path $dataset_path --output_dir $output_dir