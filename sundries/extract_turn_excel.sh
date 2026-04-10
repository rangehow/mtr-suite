#!/bin/bash

# run_summary_generator.sh
# PythonJSONExcel


# Output file
INPUT_DATA_DIR='${MTR_ROOT:-/path/to/mtr}/tempfile/eval_results/each_turn'
OUTPUT_EXCEL_FILE='turn.xlsx'

# Python (shellPATH)
PYTHON_SCRIPT_NAME="extract_turn_excel.py"

# Python
if [ ! -f "$PYTHON_SCRIPT_NAME" ]; then
 echo ": Python '$PYTHON_SCRIPT_NAME' not found"
 echo "shellor"
    exit 1
fi

#
if [ ! -d "$INPUT_DATA_DIR" ]; then
 echo ": '$INPUT_DATA_DIR' "
    exit 1
fi

echo "..."
echo ": $INPUT_DATA_DIR"
echo "Excel: $OUTPUT_EXCEL_FILE"

# Python
python "$PYTHON_SCRIPT_NAME" --input-dir "$INPUT_DATA_DIR" --output-excel "$OUTPUT_EXCEL_FILE"

# Python
if [ $? -eq 0 ]; then
 echo "Excel: $OUTPUT_EXCEL_FILE"
else
 echo "Python"
fi

exit 0