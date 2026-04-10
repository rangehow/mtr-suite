import os
import json
import pandas as pd
import glob
import argparse # For command-line arguments

def create_recall_summary_excel(folder_path, output_excel_name):
 """
 JSON, recall@20 Excel.

:
 folder_path (str): JSON.
 output_excel_name (str): Excel.
 """
 all_data_for_df = {} # {model_name: {round: recall@20}}

 # JSON
 #.json,:
 # files_to_process = glob.glob(os.path.join(folder_path, "*.json"))
 
 print(f"Scanning folder: {folder_path}")
 files_in_dir = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

 if not files_in_dir:
 print(f" '{folder_path}'.")
 return

 parsed_file_count = 0
 for filename in files_in_dir:
 filepath = os.path.join(folder_path, filename)
 # (.json,,)
 model_name = os.path.splitext(filename)[0]
 
 try:
 with open(filepath, 'r', encoding='utf-8') as f:
 data = json.load(f)
 
 model_recalls = {} # {round: recall@20}
 for round_key, metrics in data.items():
 if isinstance(metrics, dict) and "recall@20" in metrics:
 try:
 round_num = int(round_key) 
 recall_value = metrics["recall@20"]
 model_recalls[round_num] = recall_value
 except ValueError:
 print(f": '{filename}' '{round_key}'..")
 except KeyError:
 print(f": '{filename}' '{round_key}' 'recall@20'..")
 # else:
 # 
 # print(f": '{filename}' '{round_key}' recall@20.")

 if model_recalls:
 all_data_for_df[model_name] = model_recalls
 parsed_file_count += 1
 print(f"Processed '{filename}' (model: {model_name}) - found {len(model_recalls)} rounds.")
 # else:
 # print(f": '{filename}' recall@20.")

 except json.JSONDecodeError:
 print(f": '{filename}' JSON..")
 except IsADirectoryError:
 print(f": '{filename}',.") # Should not happen with os.path.isfile check
 except Exception as e:
 print(f" '{filename}': {e}..")

 if not all_data_for_df:
 print(" Excel.")
 return

 print(f"\nCollected data for {len(all_data_for_df)} models.")
 df = pd.DataFrame(all_data_for_df)
 df = df.sort_index() # Sort by round (row index)
 df.index.name = "Round"
 
 # Sort columns (model names) alphabetically for consistency
 df = df.reindex(sorted(df.columns), axis=1)
 
 try:
 df.to_excel(output_excel_name)
 print(f" Excel: '{output_excel_name}', {parsed_file_count}.")
 except Exception as e:
 print(f" Excel '{output_excel_name}' error: {e}")

if __name__ == "__main__":
 parser = argparse.ArgumentParser(description=" JSON recall@20 Excel.")
 parser.add_argument(
 "--input-dir", 
 type=str, 
 required=True, 
 help=" JSON ()."
)
 parser.add_argument(
 "--output-excel", 
 type=str, 
 required=True, 
 help=" Excel (: summary.xlsx)."
)
 
 args = parser.parse_args()
 
 create_recall_summary_excel(args.input_dir, args.output_excel)