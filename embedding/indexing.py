import numpy as np
import faiss
import os
from tqdm import tqdm
import argparse
import concurrent.futures
import logging
import shutil

# 
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(levelname)s - %(message)s'
)



def create_directory(directory_path):
 if not os.path.exists(directory_path):
 os.makedirs(directory_path)
 logging.info(f" {directory_path}.")
 else:
 logging.info(f" {directory_path}.")



def initialize_faiss_index(embedding_dim: int) -> faiss.Index:
 index = faiss.IndexFlatL2(embedding_dim)
 # index = faiss.IndexPQ(embedding_dim,16,16)
 return index

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
from shared.faiss_utils import make_vres_vdev, move_index_to_gpu




# 
def search_nearest_neighbors(vectors, index, top_k, per_query_batch_size):
 total_batches = (len(vectors) + per_query_batch_size - 1) // per_query_batch_size

 result =[]
 with tqdm(total=total_batches, desc="Searching nearest neighbors", leave=False) as pbar:
 for _, i in enumerate(range(0, len(vectors), per_query_batch_size)):
 batch = vectors[i:i+per_query_batch_size]
 _, indices = index.search(batch, top_k)
 # numpy 
 batch_results = indices.astype(np.int32)
 # 
 result.append(batch_results)
 pbar.update(1)
 
 return np.vstack(result)


# 
def process_embeddings(embeddings, ngpus, gpu_resources, top_k, per_query_batch_size, embedding_dim, output_dir,model_name):
 
 # FAISS 
 index = initialize_faiss_index(embedding_dim)
 index = move_index_to_gpu(index, ngpus, gpu_resources)
 # index.train(embeddings)
 index.add(embeddings)

 logging.info(f". {index.ntotal}.")

 nearest_neighbor_indices = search_nearest_neighbors(
 embeddings,
 index,
 top_k=top_k,
 per_query_batch_size=per_query_batch_size,
)

 # --- FIX: ---
 final_output_file = os.path.join(output_dir, f"{model_name}_nearest_neighbors.npy")
 logging.info(f": {final_output_file}")
 np.save(final_output_file, nearest_neighbor_indices) # np.save 

 logging.info(".")
 return nearest_neighbor_indices



def get_neighbors_from_faiss_index(topk,query_batch_size,index_output_dir,embeddings,model_name):

 # output_file = os.path.join(index_output_dir, f"{model_name}_nearest_neighbors.npy")
 # if os.path.exists(output_file):
 # print(f"Found existing embedding file: {output_file}")
 # print("Loading embeddings...")
 # try:
 # results = np.load(output_file, allow_pickle=True) # allow_pickle=True 
 # print("Embeddings loaded successfully.")
 # return results
 # except Exception as e:
 # print(f"load file {output_file} error: {e}")
 # print("Will recompute embeddings.")
 # # If loading fails, continue with computation

 # FAISS GPU 
 ngpus = faiss.get_num_gpus()
 logging.info(f"Number of GPUs: {ngpus}")
 
 gpu_resources = []
 for _ in range(ngpus):
 res = faiss.StandardGpuResources()
 gpu_resources.append(res)

 # 
 create_directory(index_output_dir)

 
 nearest_neighbor_indices = process_embeddings(
 embeddings=embeddings, 
 ngpus=ngpus, 
 gpu_resources=gpu_resources, 
 top_k=topk, 
 per_query_batch_size=query_batch_size, 
 embedding_dim=embeddings.shape[-1],
 output_dir=index_output_dir,
 model_name=model_name
)
 return nearest_neighbor_indices


if __name__ == "__main__":
 ngpus = faiss.get_num_gpus()
 logging.info(f"Number of GPUs: {ngpus}")
