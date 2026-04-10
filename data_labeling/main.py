import argparse
import pandas as pd
from datasets import load_from_disk, DatasetDict, Dataset
from scipy.stats import spearmanr, pearsonr, skew, kurtosis
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
 DataFrame, split, None, None, None.
 """
 dataset_short_name = os.path.basename(hf_path)
 print(f" HF: {dataset_short_name} (: {hf_path})")
 try:
 loaded_data = load_from_disk(hf_path)
 dataset_to_process = None
 split_used = "N/A"

 if isinstance(loaded_data, DatasetDict):
 found_split = False
 for split_name in preferred_splits:
 if split_name in loaded_data:
 dataset_to_process = loaded_data[split_name]
 split_used = split_name
 print(f" '{dataset_short_name}': split '{split_used}'")
 found_split = True
 break
 if not found_split:
 if loaded_data: 
 first_split_name = list(loaded_data.keys())[0]
 dataset_to_process = loaded_data[first_split_name]
 split_used = first_split_name
 print(f": '{dataset_short_name}' split. split: '{split_used}'")
 else:
 print(f": '{dataset_short_name}' DatasetDict.")
 return None, None, None
 elif isinstance(loaded_data, Dataset):
 dataset_to_process = loaded_data
 split_used = getattr(dataset_to_process, 'split', 'main') 
 else:
 print(f": '{hf_path}' Dataset DatasetDict (: {type(loaded_data)}).")
 return None, None, None

 hf_df = dataset_to_process.to_pandas()
 
 required_cols = ['faithful_score', 'quality_score']
 missing_cols = [col for col in required_cols if col not in hf_df.columns]
 if missing_cols:
 print(f": '{dataset_short_name}' (split: {split_used}): {missing_cols}")
 return None, None, None
 
 return hf_df[required_cols], dataset_short_name, split_used

 except Exception as e:
 print(f": HF '{hf_path}' error: {e}")
 import traceback
 traceback.print_exc(file=sys.stderr) 
 return None, None, None

 
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
 DataFrame, split, None, None, None.
 """
 dataset_short_name = os.path.basename(hf_path)
 print(f" HF: {dataset_short_name} (: {hf_path})")
 try:
 loaded_data = load_from_disk(hf_path)
 dataset_to_process = None
 split_used = "N/A"

 if isinstance(loaded_data, DatasetDict):
 found_split = False
 for split_name in preferred_splits:
 if split_name in loaded_data:
 dataset_to_process = loaded_data[split_name]
 split_used = split_name
 print(f" '{dataset_short_name}': split '{split_used}'")
 found_split = True
 break
 if not found_split:
 if loaded_data: 
 first_split_name = list(loaded_data.keys())[0]
 dataset_to_process = loaded_data[first_split_name]
 split_used = first_split_name
 print(f": '{dataset_short_name}' split. split: '{split_used}'")
 else:
 print(f": '{dataset_short_name}' DatasetDict.")
 return None, None, None
 elif isinstance(loaded_data, Dataset):
 dataset_to_process = loaded_data
 split_used = getattr(dataset_to_process, 'split', 'main') 
 else:
 print(f": '{hf_path}' Dataset DatasetDict (: {type(loaded_data)}).")
 return None, None, None

 hf_df = dataset_to_process.to_pandas()
 
 required_cols = ['faithful_score', 'quality_score']
 missing_cols = [col for col in required_cols if col not in hf_df.columns]
 if missing_cols:
 print(f": '{dataset_short_name}' (split: {split_used}): {missing_cols}")
 return None, None, None
 
 return hf_df[required_cols], dataset_short_name, split_used

 except Exception as e:
 print(f": HF '{hf_path}' error: {e}")
 import traceback
 traceback.print_exc(file=sys.stderr) 
 return None, None, None

def analyze_and_visualize_distributions(human_scores, llm_scores, score_type, score_range, plot_dir):
 """
 LLM.
 """
 print(f"\n--- ('{score_type}') ---")
 
 # 
 human_scores_clean = human_scores.dropna()
 llm_scores_clean = llm_scores.dropna()
 
 if len(human_scores_clean) < 2 or len(llm_scores_clean) < 2:
 print(":,.")
 return


 print(":")
 print(f" | (CSV) | LLM")
 print(f" ----------------|----------------|----------------")
 print(f" (Std Dev) | {human_scores_clean.std():<14.4f} | {llm_scores_clean.std():<14.4f}")
 print(f" (Skewness) | {skew(human_scores_clean):<14.4f} | {skew(llm_scores_clean):<14.4f}")
 print(f" (Kurtosis) | {kurtosis(human_scores_clean):<14.4f} | {kurtosis(llm_scores_clean):<14.4f}")
 print("-" * 50)


 min_score, max_score = score_range
 human_min_freq = (human_scores_clean == min_score).mean()
 human_max_freq = (human_scores_clean == max_score).mean()
 llm_min_freq = (llm_scores_clean == min_score).mean()
 llm_max_freq = (llm_scores_clean == max_score).mean()
 
 print(":")
 print(f" | (CSV) | LLM")
 print(f" ----------------|----------------|----------------")
 print(f" ({min_score}) | {human_min_freq:<14.2%} | {llm_min_freq:<14.2%}")
 print(f" ({max_score}) | {human_max_freq:<14.2%} | {llm_max_freq:<14.2%}")
 print("-" * 50)


 plt.figure(figsize=(12, 7))
 sns.histplot(human_scores_clean, color="skyblue", label="Human (CSV) Scores", kde=True, stat="density", linewidth=0)
 sns.histplot(llm_scores_clean, color="salmon", label="Aggregated LLM Scores", kde=True, stat="density", linewidth=0)
 
 plt.title(f'Score Distribution Comparison for "{score_type}"', fontsize=16)
 plt.xlabel("Score", fontsize=12)
 plt.ylabel("Density", fontsize=12)
 plt.legend()
 plt.grid(axis='y', alpha=0.5)

 if not os.path.exists(plot_dir):
 os.makedirs(plot_dir)
 
 plot_filename = os.path.join(plot_dir, f"score_distribution_{score_type.lower()}.png")
 plt.savefig(plot_filename)
 plt.close()
 
 print(f": {plot_filename}")

def main():
 parser = argparse.ArgumentParser(description=" HF, CSV.")
 parser.add_argument("csv_file_path", type=str, help="CSV.")
 parser.add_argument("hf_dataset_paths", type=str, nargs='+', help=" Hugging Face.")
 parser.add_argument("--hf_preferred_splits", type=str, nargs='+', default=['train', 'validation', 'test'],
 help=" HF DatasetDict, split.")
 parser.add_argument("--plot-dir", type=str, default="plots", help=".")
 parser.add_argument("--score-range", type=int, nargs=2, default=[1, 5], metavar=('MIN', 'MAX'),
 help=" (: 1 5).")


 args = parser.parse_args()

 #... [ 1 2]...
 # 1. CSV 
 print(f"--- CSV: {args.csv_file_path} ---")
 try:
 df_csv = pd.read_csv(args.csv_file_path)
 csv_cols_needed = ['Faithful', 'Quality']
 if not all(col in df_csv.columns for col in csv_cols_needed):
 print(f": CSV '{args.csv_file_path}': {csv_cols_needed}")
 sys.exit(1)
 
 df_csv['Faithful_numeric'] = pd.to_numeric(df_csv['Faithful'], errors='coerce')
 df_csv['Quality_numeric'] = pd.to_numeric(df_csv['Quality'], errors='coerce')
 num_rows_csv = len(df_csv)
 print(f"CSV '{os.path.basename(args.csv_file_path)}'.: {df_csv.shape} (: {num_rows_csv})")

 except Exception as e:
 print(f": CSV '{args.csv_file_path}' error: {e}")
 sys.exit(1)

 # 2. HF,, 
 print(f"\n---, HF ---")
 all_hf_faithful_scores_list = []
 all_hf_quality_scores_list = []
 successfully_processed_hf_count = 0

 for hf_path in args.hf_dataset_paths:
 scores_df, hf_short_name, hf_split_used = get_scores_from_hf_dataset(hf_path, preferred_splits=args.hf_preferred_splits)
 
 if scores_df is None: # Error occurred, message already printed by get_scores_from_hf_dataset
 print(f", '{os.path.basename(hf_path)}'.")
 continue

 if len(scores_df)!= num_rows_csv:
 print(f": '{hf_short_name}' (split: {hf_split_used}) {len(scores_df)}, CSV {num_rows_csv}..")
 continue
 
 # 
 current_hf_faithful_numeric = pd.to_numeric(scores_df['faithful_score'], errors='coerce')
 current_hf_quality_numeric = pd.to_numeric(scores_df['quality_score'], errors='coerce')

 # ---: HF CSV ---
 print(f"\n--- ( HF '{hf_short_name}' vs CSV) ---")
 section_name_prefix = f" _{hf_short_name}_vs_CSV"
 calculate_and_print_correlations(
 current_hf_faithful_numeric,
 df_csv['Faithful_numeric'],
 f"HF '{hf_short_name}' faithful_score",
 "CSV 'Faithful'",
 dataset_name=section_name_prefix
)
 calculate_and_print_correlations(
 current_hf_quality_numeric,
 df_csv['Quality_numeric'],
 f"HF '{hf_short_name}' quality_score",
 "CSV 'Quality'",
 dataset_name=section_name_prefix
)

 all_hf_faithful_scores_list.append(current_hf_faithful_numeric)
 all_hf_quality_scores_list.append(current_hf_quality_numeric)
 successfully_processed_hf_count += 1
 print(f" '{hf_short_name}' (split: {hf_split_used}).")


 if successfully_processed_hf_count == 0:
 print("\n: HF CSV..")
 sys.exit(1)
 
 print(f"\n {successfully_processed_hf_count} HF ( CSV {num_rows_csv}),.")

 # 3. HF 
 print("\n--- HF () ---")
 df_all_faithful = pd.concat(all_hf_faithful_scores_list, axis=1)
 df_all_quality = pd.concat(all_hf_quality_scores_list, axis=1)
 avg_hf_faithful_score_series = df_all_faithful.mean(axis=1)
 avg_hf_quality_score_series = df_all_quality.mean(axis=1)
 
 df_aggregated_hf_scores = pd.DataFrame({
 'average_hf_faithful_score': avg_hf_faithful_score_series,
 'average_hf_quality_score': avg_hf_quality_score_series
 })


 print("\n--- () ---")
 avg_csv_faithful = df_csv['Faithful_numeric'].mean() 
 avg_csv_quality = df_csv['Quality_numeric'].mean()
 print(f"CSV 'Faithful': {avg_csv_faithful:.4f} ( {df_csv['Faithful_numeric'].notna().sum()})")
 print(f"CSV 'Quality': {avg_csv_quality:.4f} ( {df_csv['Quality_numeric'].notna().sum()})")

 avg_agg_hf_faithful = df_aggregated_hf_scores['average_hf_faithful_score'].mean()
 avg_agg_hf_quality = df_aggregated_hf_scores['average_hf_quality_score'].mean()
 print(f" HF 'faithful_score': {avg_agg_hf_faithful:.4f} ( {df_aggregated_hf_scores['average_hf_faithful_score'].notna().sum()})")
 print(f" HF 'quality_score': {avg_agg_hf_quality:.4f} ( {df_aggregated_hf_scores['average_hf_quality_score'].notna().sum()})")


 # 5. HF CSV 
 print("\n--- ( HF vs CSV) ---")
 calculate_and_print_correlations(
 df_aggregated_hf_scores['average_hf_faithful_score'],
 df_csv['Faithful_numeric'],
 " HF 'average_faithful_score'",
 "CSV 'Faithful'",
 dataset_name=" HF_vs_CSV"
)

 calculate_and_print_correlations(
 df_aggregated_hf_scores['average_hf_quality_score'],
 df_csv['Quality_numeric'],
 " HF 'average_quality_score'",
 "CSV 'Quality'",
 dataset_name=" HF_vs_CSV"
)

 # --- 6.: ---
 analyze_and_visualize_distributions(
 human_scores=df_csv['Faithful_numeric'],
 llm_scores=df_aggregated_hf_scores['average_hf_faithful_score'],
 score_type='Faithful',
 score_range=args.score_range,
 plot_dir=args.plot_dir
)
 analyze_and_visualize_distributions(
 human_scores=df_csv['Quality_numeric'],
 llm_scores=df_aggregated_hf_scores['average_hf_quality_score'],
 score_type='Quality',
 score_range=args.score_range,
 plot_dir=args.plot_dir
)

 print("\nPython.")

if __name__ == "__main__":
 try:
 import pandas, datasets, scipy, numpy, matplotlib, seaborn
 except ImportError as e:
 print(f": Python. pandas, datasets, scipy, numpy, matplotlib, seaborn.")
 print(f": pip install pandas datasets scipy numpy matplotlib seaborn pyarrow")
 print(f": {e}")
 sys.exit(1)
 main()