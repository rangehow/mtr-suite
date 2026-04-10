import os
import json
import pandas as pd
import argparse

def main():
 parser = argparse.ArgumentParser(description="Batch convert JSON result files to an Excel spreadsheet.")

 parser.add_argument(
 "-i", "--input-dir",
 type=str,
 default=".",
 help="Directory containing the JSON result files. Defaults to the current directory (.)."
)
 parser.add_argument(
 "-o", "--output-file",
 type=str,
 default="model_dataset_results.xlsx",
 help="Name of the output Excel file. Defaults to 'model_dataset_results.xlsx'."
)
 parser.add_argument(
 "-p", "--file-prefix",
 type=str,
 default="results_",
 help="Prefix of the JSON result files to process. Defaults to 'results_'."
)
 parser.add_argument(
 "-s", "--file-suffix",
 type=str,
 default=".json",
 help="Suffix of the JSON result files to process. Defaults to '.json'."
)

 # 
 args = parser.parse_args()

 json_directory = args.input_dir
 output_excel_file = args.output_file
 file_prefix = args.file_prefix
 file_suffix = args.file_suffix

 data_list = []
 processed_files_count = 0

 print(f":")
 print(f": '{json_directory}'")
 print(f": '{output_excel_file}'")
 print(f": '{file_prefix}'")
 print(f": '{file_suffix}'")
 print("-" * 20)

 if not os.path.isdir(json_directory):
 print(f": '{json_directory}'.")
 return # 

 print(f" '{json_directory}' JSON...")

 # 
 for filename in os.listdir(json_directory):
 # 
 if filename.startswith(file_prefix) and filename.endswith(file_suffix):
 file_path = os.path.join(json_directory, filename)

 # 
 if os.path.isfile(file_path):
 print(f": {filename}")
 processed_files_count += 1
 try:
 # JSON 
 with open(file_path, 'r', encoding='utf-8') as f:
 data = json.load(f)

 # 
 extracted_data = {
 'model_name': data.get('model_name'),
 'dataset_name': data.get('dataset_name'),
 'recall@20': data.get('recall@20'),
 'recall@5': data.get('recall@5'),
 'mrr@20': data.get('mrr@20'),
 'mrr@5': data.get('mrr@5'),
 'ndcg@20': data.get('ndcg@20'),
 'ndcg@5': data.get('ndcg@5'),
 # JSON, 
 # data.get('new_metric_name'): data.get('new_metric_name'),
 }
 data_list.append(extracted_data)

 except json.JSONDecodeError:
 print(f": {filename} JSON,.")
 except FileNotFoundError:
 # Should not happen after os.listdir and os.path.isfile checks
 print(f": {filename},.")
 except Exception as e:
 print(f" {filename}: {e},.")

 print(f", {processed_files_count}.")

 if not data_list:
 print(" JSON Excel.")
 else:
 # Pandas DataFrame
 df = pd.DataFrame(data_list)

 desired_column_order = [
 'model_name',
 'dataset_name',
 'recall@20',
 'recall@5',
 'mrr@20',
 'mrr@5',
 'ndcg@20',
 'ndcg@5'
]
 # ( get,)
 existing_columns = [col for col in desired_column_order if col in df.columns]
 df = df[existing_columns]


 # DataFrame Excel 
 try:
 df.to_excel(output_excel_file, index=False) # index=False DataFrame Excel
 print(f" Excel: {output_excel_file}")
 except Exception as e:
 print(f" Excel: {e}")

if __name__ == "__main__":
 main()