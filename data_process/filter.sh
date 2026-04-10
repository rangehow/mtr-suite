

unfiltered_dataset="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/processed_finance"
filtered_save_path="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/processed_finance_filtered"
sample_save_path="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/processed_finance_unfiltered"

python ${MTR_ROOT:-/path/to/mtr}/data_process/final_filter.py --dataset_path ${unfiltered_dataset} --save_path ${filtered_save_path} --sample_save_path ${sample_save_path}