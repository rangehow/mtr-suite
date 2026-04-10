import argparse
import pandas as pd
from datasets import load_from_disk, DatasetDict, Dataset
from scipy.stats import spearmanr, pearsonr
import sys
import os
import numpy as np

def calculate_and_print_correlations(series1, series2, name1, name2, dataset_name=""):
 """ Series."""
 prefix = f"[{dataset_name}] " if dataset_name else ""

 if series1.empty or series2.empty:
 print(f"{prefix}: Series ('{name1}' len: {len(series1)}, '{name2}' len: {len(series2)})..")
 return

 if len(series1)!= len(series2):
 print(f"{prefix}: '{name1}' ( {len(series1)}) '{name2}' ( {len(series2)})..")
 return

 s1_numeric = pd.to_numeric(series1, errors='coerce')
 s2_numeric = pd.to_numeric(series2, errors='coerce')

 valid_s1 = s1_numeric.notna()
 valid_s2 = s2_numeric.notna()
 combined_valid = valid_s1 & valid_s2

 s1_final = s1_numeric[combined_valid]
 s2_final = s2_numeric[combined_valid]
 
 num_valid_samples = len(s1_final)

 if num_valid_samples < 2:
 print(f"{prefix}:,'{name1}' '{name2}' ({num_valid_samples}) 2..")
 return

 spearman_corr, spearman_p = np.nan, np.nan # Initialize with NaN
 pearson_corr, pearson_p = np.nan, np.nan

 try:
 spearman_corr, spearman_p = spearmanr(s1_final, s2_final)
 print(f"{prefix} (Spearman) - '{name1}' vs '{name2}': {spearman_corr:.4f} (p-value: {spearman_p:.4e}, {num_valid_samples})")
 except Exception as e:
 print(f"{prefix} '{name1}' vs '{name2}': {e}")

 try:
 if len(s1_final.unique()) < 2 or len(s2_final.unique()) < 2:
 print(f"{prefix} (Pearson) - '{name1}' vs '{name2}':, ( {num_valid_samples}).")
 else:
 pearson_corr, pearson_p = pearsonr(s1_final, s2_final)
 print(f"{prefix} (Pearson) - '{name1}' vs '{name2}': {pearson_corr:.4f} (p-value: {pearson_p:.4e}, {num_valid_samples})")
 except Exception as e:
 print(f"{prefix} '{name1}' vs '{name2}': {e}")

def get_scores_from_hf_dataset(hf_path, preferred_splits=['train', 'validation', 'test']):
 """
 HF, 'faithful_score' 'quality_score'.
 DatasetDict, split.
 DataFrame, None.
 """
 dataset_short_name = os.path.basename(hf_path)
 print(f" HF: {dataset_short_name} (: {hf_path})")
 try:
 loaded_data = load_from_disk(hf_path)
 dataset_to_process = None
 split_used = "N/A"

 if isinstance(loaded_data, DatasetDict):
 # print(f" '{dataset_short_name}' DatasetDict. splits: {preferred_splits}")
 found_split = False
 for split_name in preferred_splits:
 if split_name in loaded_data:
 dataset_to_process = loaded_data[split_name]
 split_used = split_name
 print(f" '{dataset_short_name}': split '{split_used}'")
 found_split = True
 break
 if not found_split:
 if loaded_data: # splits preferred
 first_split_name = list(loaded_data.keys())[0]
 dataset_to_process = loaded_data[first_split_name]
 split_used = first_split_name
 print(f": '{dataset_short_name}' split. split: '{split_used}'")
 else:
 print(f": '{dataset_short_name}' DatasetDict.")
 return None
 elif isinstance(loaded_data, Dataset):
 # print(f" '{dataset_short_name}' Dataset.")
 dataset_to_process = loaded_data
 split_used = getattr(dataset_to_process, 'split', 'main') # Try to get split name if available
 else:
 print(f": '{hf_path}' Dataset DatasetDict (: {type(loaded_data)}).")
 return None

 hf_df = dataset_to_process.to_pandas()
 
 required_cols = ['tag_score']
 missing_cols = [col for col in required_cols if col not in hf_df.columns]
 if missing_cols:
 print(f": '{dataset_short_name}' (split: {split_used}): {missing_cols}")
 return None
 
 return hf_df[required_cols]

 except Exception as e:
 print(f": HF '{hf_path}' error: {e}")
 import traceback
 traceback.print_exc(file=sys.stderr) # Print full traceback for debugging
 return None

def main():
 parser = argparse.ArgumentParser(description=" HF, CSV.")
 parser.add_argument("csv_file_path", type=str, help="CSV.")
 parser.add_argument("hf_dataset_paths", type=str, nargs='+', help=" Hugging Face.")
 parser.add_argument("--hf_preferred_splits", type=str, nargs='+', default=['train', 'validation', 'test'],
 help=" HF DatasetDict, split.")

 args = parser.parse_args()

 # 1. CSV 
 print(f"--- CSV: {args.csv_file_path} ---")
 try:
 df_csv = pd.read_csv(args.csv_file_path)
 csv_cols_needed = ['document_correctness']
 if not all(col in df_csv.columns for col in csv_cols_needed):
 print(f": CSV '{args.csv_file_path}': {csv_cols_needed}")
 sys.exit(1)
 
 df_csv['document_correctness'] = pd.to_numeric(df_csv['document_correctness'], errors='coerce')
 num_rows_csv = len(df_csv)
 print(f"CSV '{os.path.basename(args.csv_file_path)}'.: {df_csv.shape} (: {num_rows_csv})")

 except Exception as e:
 print(f": CSV '{args.csv_file_path}' error: {e}")
 sys.exit(1)

 # 2. HF 
 print(f"\n--- {len(args.hf_dataset_paths)} HF ---")
 all_hf_faithful_scores_list = []
 all_hf_quality_scores_list = []
 successfully_processed_hf_count = 0

 for hf_path in args.hf_dataset_paths:
 scores_df = get_scores_from_hf_dataset(hf_path, preferred_splits=args.hf_preferred_splits)
 if scores_df is not None:
 # if len(scores_df)!= num_rows_csv:
 # print(f": '{os.path.basename(hf_path)}' {len(scores_df)}, CSV {num_rows_csv}..")
 # continue
 

 all_hf_quality_scores_list.append(pd.to_numeric(scores_df['tag_score'][:len(df_csv)], errors='coerce'))
 successfully_processed_hf_count += 1
 else:
 print(f", '{os.path.basename(hf_path)}'.")

 if successfully_processed_hf_count == 0:
 print("\n: HF CSV..")
 sys.exit(1)
 
 print(f"\n {successfully_processed_hf_count} HF ( CSV {num_rows_csv}).")

 # 3. HF 
 print("\n--- HF ---")
 # Series DataFrame, 
 #.mean(axis=1) NaN (skipna=True)
 df_all_tag = pd.concat(all_hf_quality_scores_list, axis=1)

 df_all_tag.columns = [f"tag_hf_{i}" for i in range(df_all_tag.shape[1])]

 avg_hf_tag_score_series = df_all_tag.mean(axis=1)

 
 print(f" 'faithful_score' (: {len(avg_hf_tag_score_series)})")

 df_aggregated_hf_scores = pd.DataFrame({
 'avg_hf_tag_score_series': avg_hf_tag_score_series,

 })


 print("\n--- ---")
 # CSV 
 avg_csv_tag = df_csv['document_correctness'].mean() # mean() NaN
 
 print(f"CSV 'Quality': {avg_csv_tag:.4f} ( {df_csv['document_correctness'].notna().sum()})")

 # HF 
 avg_agg_hf_tag = df_aggregated_hf_scores['avg_hf_tag_score_series'].mean()
 print(f" HF 'faithful_score': {avg_agg_hf_tag:.4f} ( {df_aggregated_hf_scores['avg_hf_tag_score_series'].notna().sum()})")
 # 5. HF CSV 
 print("\n--- ( HF vs CSV) ---")
 calculate_and_print_correlations(
 df_aggregated_hf_scores['avg_hf_tag_score_series'],
 df_csv['document_correctness'],
 " HF 'average_faithful_score'",
 "CSV 'Faithful'",
 dataset_name=" HF_vs_CSV"
)



 print("\nPython.")

if __name__ == "__main__":
 try:
 import pandas
 import datasets
 import scipy
 import numpy
 except ImportError as e:
 print(f": Python. pandas, datasets, scipy, numpy.")
 print(f": pip install pandas datasets scipy numpy pyarrow")
 print(f": {e}")
 sys.exit(1)
 main()