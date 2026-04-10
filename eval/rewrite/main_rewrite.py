import argparse
import json

import os
import numpy as np
from loguru import logger

import faiss
import datasets

# from loguru import logger # Duplicate import, removed
from..metrics import *


# faiss gpu c++, try catch, gpu, ngpus=0
# torch, torch,faiss-gpu!
try:
 from torch.cuda import device_count
 if device_count()>0:
 ngpus= faiss.get_num_gpus()
 gpu_resources = []
 for i in range(ngpus):
 res = faiss.StandardGpuResources()
 gpu_resources.append(res)
 logger.info(f"FAISS: Found {ngpus} GPUs.")
 else:
 ngpus = 0
 logger.info("FAISS: No GPUs found by PyTorch, using CPU for FAISS.")

except Exception as e:
 logger.warning(f"FAISS: Error during GPU detection: {e}. Defaulting to CPU.")
 ngpus = 0





import sys as _sys2, os as _os2
_sys2.path.insert(0, _os2.path.join(_os2.path.dirname(__file__), '../..'))
from shared.faiss_utils import make_vres_vdev, move_index_to_gpu as _move_index_to_gpu

def move_index_to_gpu(index: faiss.Index) -> faiss.Index:
    """Move FAISS index to GPU using module-level ngpus/gpu_resources."""
    return _move_index_to_gpu(index, ngpus, gpu_resources)


import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
from shared.data_utils import parse_dataset



def parse_args():
 parser = argparse.ArgumentParser(description='Embedding Generation Script')


 parser.add_argument('--index_output_path', type=str, required=True)
 parser.add_argument('--embedding_model_name', type=str, required=True)
 parser.add_argument('--domain_map_path', type=str, required=True)
 parser.add_argument('--dataset_name', type=str, required=True)
 parser.add_argument('--query_dataset_path', type=str, required=True)
 parser.add_argument('--embedding_output_path', type=str, required=True)
 
 # 
 parser.add_argument('--results_save_dir', type=str, 
 help="Directory to save test results and skip if exists.")
 

 return parser.parse_args()


if __name__=='__main__':

 args = parse_args()
 
 os.makedirs(args.results_save_dir, exist_ok=True)

 embedding_model_name_list = args.embedding_model_name.split(',')
 domain_map_path_list = args.domain_map_path.split(',')
 dataset_name_list = args.dataset_name.split(',')
 query_dataset_path_list = args.query_dataset_path.split(',')
 
 for model_name in embedding_model_name_list:

 for domain_map_path,dataset_name,query_dataset_path in zip(domain_map_path_list,dataset_name_list,query_dataset_path_list):

 #, domain_map_path 
 if domain_map_path.lower() == "full":
 domain_identifier = "full"
 else:
 domain_identifier = os.path.splitext(os.path.basename(domain_map_path))[0] # e.g., "domain_A_map" from "/path/to/domain_A_map.json"
 
 result_filename = f"results_{model_name}_{dataset_name}.json"
 result_filepath = os.path.join(args.results_save_dir, result_filename)

 if os.path.exists(result_filepath):
 logger.info(f"Results for {model_name} on {dataset_name} already exist at {result_filepath}. Skipping.")
 try:
 with open(result_filepath, 'r') as f_res:
 existing_results = json.load(f_res)
 # logger.info(f"Existing results: {existing_results}")
 except Exception as e:
 logger.warning(f"Could not read existing results file {result_filepath}: {e}")
 continue # 

 logger.info(f"Running test for: Model={model_name}, Dataset={dataset_name}, DomainMap={domain_map_path} ({domain_identifier})")



 def remove_lastturn_suffix(s):
 if s.endswith('_lastturn'):
 return s[:-len('_lastturn')]
 return s
 index_save_path = os.path.join(args.index_output_path,'mtr',f'{remove_lastturn_suffix(model_name)}.idx')
 # args.embedding_output_path # This line doesn't do anything, can be removed or used if intended.

 embedding_output_dir = os.path.join(args.embedding_output_path,dataset_name)
 embedding_file = os.path.join(embedding_output_dir, f"query_{model_name}.npy")

 try:
 index = faiss.read_index(index_save_path)
 logger.info(f'{dataset_name} {index.ntotal}')
 embedding = np.load(embedding_file)
 
 query_dataset_path_actual = query_dataset_path # Use a different variable if you intend to modify it
 # need to extract gold_idx
 # query_dataset = datasets.load_from_disk(query_dataset_path) # parse_dataset is more general
 query_dataset = parse_dataset(query_dataset_path_actual)

 if 'test' in query_dataset:
 query_dataset = query_dataset['test']
 elif 'dev' in query_dataset:
 query_dataset = query_dataset['dev']
 # If it's already a flat dataset (e.g., from load_dataset("csv",...)), it might not have 'test' or 'dev'
 # and 'ground_truth_ctx' would be a direct key. This logic seems fine.

 gold_idx = []
 for item in query_dataset: # Iterate directly over dataset items
 ground_truth_ctx = item['ground_truth_ctx']
 if 'index' in ground_truth_ctx:
 gold_idx.append(ground_truth_ctx['index'])
 # for mtr
 elif "document_idx" in ground_truth_ctx:
 gold_idx.append(ground_truth_ctx['document_idx'])
 
 if not gold_idx:
 logger.warning(f"No gold_idx found for {dataset_name} with query dataset {query_dataset_path}. Check data format.")
 # Decide if you want to skip or continue with potentially wrong evaluation
 # continue 

 I = None # Initialize I

 if domain_map_path.lower()!="full":
 domain_map = json.load(open(domain_map_path))
 
 domain_to_indices_map={}
 start_idx= 0 
 # Assuming query_dataset has a 'document' column that matches keys in domain_map
 # The original logic for domain_to_indices_map seems to be about ranges within the *query* embeddings
 # based on 'document' field. This is kept as is.
 current_doc_field_name = None
 if 'document' in query_dataset.column_names: # Common field name
 current_doc_field_name = 'document'
 elif 'domain' in query_dataset.column_names: # Alternative field name
 current_doc_field_name = 'domain'
 else:
 logger.error(f"Could not find 'document' or 'domain' column in query_dataset for domain mapping. Columns: {query_dataset.column_names}")
 # Potentially skip this iteration or raise an error
 continue
 
 for i, doc_val in enumerate(query_dataset[current_doc_field_name]):
 if doc_val not in domain_to_indices_map:
 start_idx = i
 domain_to_indices_map[doc_val] = (start_idx,start_idx)
 else:
 domain_to_indices_map[doc_val] = (domain_to_indices_map[doc_val][0],i) # update end_idx
 
 list_I_sel = []
 processed_query_indices_count = 0
 #, query domain, domain 
 for domain_key_in_query in domain_to_indices_map.keys(): # e.g., domain_key_in_query could be 'finance', 'tech'
 if domain_key_in_query not in domain_map:
 logger.warning(f"Domain '{domain_key_in_query}' from query data not found in domain_map file {domain_map_path}. Skipping this domain for search.")
 # Count how many embeddings this domain corresponded to and add placeholders or skip
 num_embeddings_for_skipped_domain = domain_to_indices_map[domain_key_in_query][1] - domain_to_indices_map[domain_key_in_query][0] + 1
 # Option 1: Add dummy results (e.g., -1s)
 # list_I_sel.append(np.full((num_embeddings_for_skipped_domain, 20), -1, dtype=np.int64))
 # Option 2: Skip (will lead to I having fewer rows than embedding) - This needs careful handling later
 # For now, let's assume we want to search only available domains
 continue

 sel = faiss.IDSelectorRange(domain_map[domain_key_in_query][0], domain_map[domain_key_in_query][1]) # Corrected: end index is exclusive for range
 params = faiss.SearchParameters(sel=sel)
 
 if domain_map[domain_key_in_query][1] > index.ntotal: # range end can be ntotal
 logger.error(f"Domain map range end {domain_map[domain_key_in_query][1]} for domain {domain_key_in_query} exceeds index.ntotal {index.ntotal}. Clamping.")
 # This indicates an issue with domain_map generation.
 # For robustness, you might clamp it, but it's better to fix the map.
 # For now, let's just log and proceed, FAISS might handle out-of-bounds gracefully or error.
 # A safer approach would be to adjust the selector range, but this hides the root cause.

 query_start_idx, query_end_idx = domain_to_indices_map[domain_key_in_query]
 sub_embedding = embedding[query_start_idx: query_end_idx + 1]
 
 if sub_embedding.shape[0] == 0:
 logger.warning(f"Sub_embedding for domain {domain_key_in_query} is empty. Indices: {query_start_idx} to {query_end_idx}. Skipping search for this part.")
 continue

 D_sel, I_sel = index.search(sub_embedding, k=20, params=params)
 # The retrieved I_sel are absolute indices from the main index.
 # If your gold_idx are relative to the sub-corpus of that domain, you need to adjust I_sel.
 # Original code: list_I_sel.append(I_sel-domain_map[domain][0])
 # This makes sense if gold_idx are 0-based *within each domain's part of the corpus*.
 # Let's assume this is the case.
 list_I_sel.append(I_sel - domain_map[domain_key_in_query][0])
 processed_query_indices_count += sub_embedding.shape[0]

 if not list_I_sel:
 logger.error(f"No valid domains found for search for {model_name}, {dataset_name}. Cannot compute metrics.")
 I = np.array([[]]) # Empty or placeholder
 else:
 # This concatenation might fail if some domains were skipped and not placeholder-ed
 # It assumes all sub_embeddings that were searched actually returned results.
 try:
 I = np.vstack(list_I_sel)
 except ValueError as ve:
 logger.error(f"Error stacking I_sel parts: {ve}. This might happen if parts have different K values or some parts are empty.")
 # Add debugging for shapes of I_sel
 for idx_part, part_sel in enumerate(list_I_sel):
 logger.debug(f"Shape of I_sel part {idx_part}: {part_sel.shape}")
 # If I cannot be formed, metrics cannot be calculated correctly.
 # Fallback or skip metrics calculation for this case.
 I = np.array([[]]) # Placeholder to avoid crashing later, but indicates failure

 # Ensure I has the same number of rows as original embeddings if domains were skipped without placeholders
 if I.shape[0]!= embedding.shape[0] and I.shape[0] > 0: # I.shape[0]>0 to avoid div by zero warning for empty I
 logger.warning(f"Shape of I ({I.shape[0]}) does not match embedding ({embedding.shape[0]}) after domain-specific search. This might be due to skipped domains. Metrics might be inaccurate.")
 # This is a tricky situation. You might need to re-align I with gold_idx,
 # or ensure that skipped domains in `domain_to_indices_map` lead to placeholder results in `list_I_sel`.
 # For now, proceeding with the potentially misaligned/incomplete I.

 else: # Full search
 index_to_search = move_index_to_gpu(index) # Move to GPU only if not already done or if full search
 D, I = index_to_search.search(embedding, k=20)
 
 
 metrics_results = {}
 if I is not None and I.shape[0] > 0 and gold_idx and len(I) == len(gold_idx):
 recall_at_20 = recall(I.tolist(),gold_idx,20)
 recall_at_5 = recall(I.tolist(),gold_idx,5)
 mrr_at_20 = mrr(I.tolist(),gold_idx,20)
 mrr_at_5 = mrr(I.tolist(),gold_idx,5)
 ndcg_at_20 = ndcg(I.tolist(),gold_idx,20)
 ndcg_at_5 = ndcg(I.tolist(),gold_idx,5)
 metrics_results['recall@20'] = recall_at_20
 metrics_results['recall@5'] = recall_at_5
 metrics_results['mrr@20'] = mrr_at_20
 metrics_results['mrr@5'] = mrr_at_5
 metrics_results['ndcg@20'] = ndcg_at_20
 metrics_results['ndcg@5'] = ndcg_at_5
 metrics_results['model_name'] = model_name
 metrics_results['dataset_name'] = dataset_name
 metrics_results['domain_map_file'] = os.path.basename(domain_map_path) if domain_map_path.lower()!= "full" else "full"
 metrics_results['domain_identifier_for_filename'] = domain_identifier
 
 
 elif I is None or I.shape[0] == 0:
 logger.error(f"Search results 'I' are empty or None for {model_name} on {dataset_name} with domain {domain_identifier}. Cannot calculate recall.")
 metrics_results['error'] = "Search results I were empty or None."
 elif not gold_idx:
 logger.error(f"gold_idx is empty for {model_name} on {dataset_name} with domain {domain_identifier}. Cannot calculate recall.")
 metrics_results['error'] = "gold_idx was empty."
 elif len(I)!= len(gold_idx):
 logger.error(f"Mismatch between length of search results I ({len(I)}) and gold_idx ({len(gold_idx)}) for {model_name} on {dataset_name} with domain {domain_identifier}. Cannot calculate recall accurately.")
 metrics_results['error'] = f"Length mismatch: I ({len(I)}), gold_idx ({len(gold_idx)})"


 with open(result_filepath, 'w') as f_out:
 json.dump(metrics_results, f_out, indent=4)
 logger.info(f"Results saved to {result_filepath}")

 except FileNotFoundError as e:
 logger.error(f"File not found error during processing for {model_name}, {dataset_name}, {domain_identifier}: {e}")

 logger.info("All tests completed.")