#!/bin/bash

# -e
set -e

# Python
PYTHON_SCRIPT="convert_eval_results_to_excel.py"

# Python
if [ ! -f "$PYTHON_SCRIPT" ]; then
 echo ": Python '$PYTHON_SCRIPT' not found!shell"
    exit 1
fi



# --- 2: Output file ---
echo "--- 2: Output file ---"
# JSON 'my_results_folder'
# 'summary_report.xlsx'
# : 'my_results_folder' JSON
# mkdir -p my_results_folder
# cp results_*.json my_results_folder/ #

# ()
python "$PYTHON_SCRIPT" --input-dir ${MTR_ROOT:-/path/to/mtr}/tempfile/eval_results --output-file summary_report.xlsx
echo "" #

#
# stats_*.json
# echo "--- 3: Output file ---"
# # stats_model_dataset.json
# # cp results_bge-large-en-v1.5_mtr-rewrite-7B.json stats_bge-large-en-v1.5_mtr-rewrite-7B.json #
# python "$PYTHON_SCRIPT" --file-prefix stats_ --output-file model_dataset_results_stats.xlsx
# echo ""

echo "Shell Script execution complete."