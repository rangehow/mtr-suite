import csv
import json
from tqdm import tqdm
import argparse

def main(tsv_file_path, input_json_path, output_with_index_path):
 """
 TSV JSON, ground_truth_ctx JSON.

:
 tsv_file_path (str): TSV.
 input_json_path (str): JSON.
 output_with_index_path (str): JSON.
 """

 print(f" TSV:{tsv_file_path}")
 text_list = []

 try:
 with open(tsv_file_path, 'r', encoding='utf-8') as file:
 tsv_reader = csv.reader(file, delimiter='\t')
 try:
 header = next(tsv_reader)
 except StopIteration:
 print(f":TSV {tsv_file_path}.")
 return
 
 try:
 text_index = header.index("text")
 except ValueError:
 print(f":TSV {tsv_file_path} 'text'.")
 return

 for row in tsv_reader:
 if len(row) > text_index:
 text_list.append(row[text_index])
 else:
 print(f":TSV 'text',.: {row}")

 except FileNotFoundError:
 print(f":TSV:{tsv_file_path}")
 return
 except Exception as e:
 print(f" TSV:{e}")
 return

 print(f" JSON:{input_json_path}")
 try:
 with open(input_json_path, "r", encoding="utf-8") as f:
 input_list = json.load(f)
 except FileNotFoundError:
 print(f": JSON:{input_json_path}")
 return
 except json.JSONDecodeError:
 print(f": JSON:{input_json_path}")
 return
 except Exception as e:
 print(f" JSON:{e}")
 return

 wiki_dict = {text: idx for idx, text in enumerate(text_list)}

 print(" ground_truth_ctx...")
 unmatched_count = 0

 for item in tqdm(input_list, desc="Processing input list"):
 item["document"] = "wiki"
 ground_truth_ctx = item.get("ground_truth_ctx", {})
 ctx = ground_truth_ctx.get("ctx", "")

 idx = wiki_dict.get(ctx, -1)
 ground_truth_ctx["index"] = idx
 if idx == -1:
 unmatched_count += 1

 print(f" ground_truth_ctx: {unmatched_count}")

 try:
 with open(output_with_index_path, "w", encoding="utf-8") as f:
 json.dump(input_list, f, ensure_ascii=False, indent=4)
 print(f":{output_with_index_path}")
 except Exception as e:
 print(f" JSON:{e}")

if __name__ == "__main__":
 parser = argparse.ArgumentParser(description=" JSON.")
 parser.add_argument("tsv_file", help=" TSV.")
 parser.add_argument("input_json", help=" JSON.")
 parser.add_argument("output_json", help=" JSON.")

 args = parser.parse_args()

 main(args.tsv_file, args.input_json, args.output_json)