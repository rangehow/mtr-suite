
cd ${MTR_ROOT:-/path/to/mtr}/model_choice

# result_dataset_path="${MTR_ROOT:-/path/to/mtr}/tempfile/choice_coral"

# result_dataset_path=${MTR_ROOT:-/path/to/mtr}/tempfile/finance_filtered_choice
# output_dir=${MTR_ROOT:-/path/to/mtr}/model_choice/summary_plots_finance_filtered
# result_dataset_path="${MTR_ROOT:-/path/to/mtr}/tempfile/choice"

result_dataset_path=${MTR_ROOT:-/path/to/mtr}/tempfile/finance_choice
output_dir=${MTR_ROOT:-/path/to/mtr}/model_choice/summary_plots_finance_unfiltered
python ${MTR_ROOT:-/path/to/mtr}/model_choice/analysis_result.py --result_dataset_path $result_dataset_path --output_dir $output_dir