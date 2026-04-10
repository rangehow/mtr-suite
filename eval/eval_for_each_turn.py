import argparse
from collections import defaultdict
import json

import os
import numpy as np
from loguru import logger

import faiss
import datasets

# from loguru import logger # Duplicate import, removed
from metrics import *


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
_sys2.path.insert(0, _os2.path.join(_os2.path.dirname(__file__), '..'))
from shared.faiss_utils import make_vres_vdev, move_index_to_gpu as _move_index_to_gpu

def move_index_to_gpu(index: faiss.Index) -> faiss.Index:
    """Move FAISS index to GPU using module-level ngpus/gpu_resources."""
    return _move_index_to_gpu(index, ngpus, gpu_resources)


import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
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
 
 os.makedirs(os.path.join(args.results_save_dir,'each_turn'), exist_ok=True)

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
 result_filepath = os.path.join(args.results_save_dir,'each_turn', result_filename)

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

 index_save_path = os.path.join(args.index_output_path,dataset_name,f'{model_name}.idx')
 # args.embedding_output_path # This line doesn't do anything, can be removed or used if intended.

 embedding_output_dir = os.path.join(args.embedding_output_path,dataset_name)
 embedding_file = os.path.join(embedding_output_dir, f"query_{model_name}.npy")

 query_dataset_path_actual = query_dataset_path # Use a different variable if you intend to modify it
 # need to extract gold_idx
 # query_dataset = datasets.load_from_disk(query_dataset_path) # parse_dataset is more general
 query_dataset = parse_dataset(query_dataset_path_actual)

 if 'test' in query_dataset:
 query_dataset = query_dataset['test']
 elif 'dev' in query_dataset:
 query_dataset = query_dataset['dev']

 try:
 index = faiss.read_index(index_save_path)
 logger.info(f'{dataset_name} {index.ntotal}')
 embedding = np.load(embedding_file)
 
 
 # If it's already a flat dataset (e.g., from load_dataset("csv",...)), it might not have 'test' or 'dev'
 # and 'ground_truth_ctx' would be a direct key. This logic seems fine.

 gold_idx_full = defaultdict(list)
 idx_for_embedding_slice = defaultdict(list)
 for i,item in enumerate(query_dataset): # Iterate directly over dataset items
 
 ground_truth_ctx = item['ground_truth_ctx']
 gold_idx_full[len(item['messages'])//2].append(ground_truth_ctx['document_idx'])
 idx_for_embedding_slice[len(item['messages'])//2].append(i)
 # gold_idx.append(ground_truth_ctx['document_idx'])
 I = None # Initialize I


 index_to_search = move_index_to_gpu(index) # Move to GPU only if not already done or if full search
 _, I_full = index_to_search.search(embedding, k=20)
 metrics_results_total = {}
 for turn,index in idx_for_embedding_slice.items():
 I = I_full[index]
 gold_idx = gold_idx_full[turn]
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

 metrics_results_total[turn] = metrics_results

 

 except FileNotFoundError as e:
 logger.error(f"File not found error during processing for {model_name}, {dataset_name}, {domain_identifier}: {e}")

 with open(result_filepath, 'w') as f_out:
 json.dump(metrics_results_total, f_out, indent=4)
 logger.info(f"Results saved to {result_filepath}")
 logger.info("All tests completed.")