
import os

import numpy as np
from arg_parser import parse_args # 


from functools import partial
from utils import parse_dataset

from cluster import get_cluster_dataset

def main():
 # 
 args = parse_args()
 
 dataset = parse_dataset(args.processed_dataset_path)
 
 neighbor_output_file = os.path.join(args.index_output_dir, f"{args.model_name}_nearest_neighbors.npy")

 nearest_neighbor_indices = None # 

 # ---: main FAISS ---
 if os.path.exists(neighbor_output_file):
 print(f": {neighbor_output_file}")
 print("...")
 try:
 # 
 nearest_neighbor_indices = np.load(neighbor_output_file, allow_pickle=True)
 print(".")
 #, embeddings get_neighbors_from_faiss_index 

 except Exception as e:
 print(f"load file {neighbor_output_file} error: {e}")
 print("Will recompute.")
 nearest_neighbor_indices = None #, None, 


 if nearest_neighbor_indices is None:
 
 embeddings=None
 output_file = os.path.join(args.embedding_output_dir, f"{args.model_name}.npy")

 # 2. Check if file exists
 if os.path.exists(output_file):
 print(f"Found existing embedding file: {output_file}")
 print("Loading embeddings...")
 try:
 embeddings = np.load(output_file, allow_pickle=True) # allow_pickle=True 
 print(embeddings.shape)
 print(len(dataset))
 print("Embeddings loaded successfully.")

 except Exception as e:
 print(f"load file {output_file} error: {e}")
 print("Will recompute embeddings.")
 embeddings = None

 else:
 from _vllm_embedding import get_embedding
 embeddings = get_embedding(args.model_name,dataset,args.embedding_model,args.embedding_output_dir)
 
 if embeddings is None:
 from _vllm_embedding import get_embedding
 embeddings = get_embedding(args.model_name,dataset,args.embedding_model,args.embedding_output_dir)


 from indexing import get_neighbors_from_faiss_index
 nearest_neighbor_indices = get_neighbors_from_faiss_index(args.topk,args.query_batch_size,args.index_output_dir,embeddings,args.model_name)

 cluster_dataset = get_cluster_dataset(cluster_dataset_output_dir=args.cluster_dataset_output_dir,neighbors=nearest_neighbor_indices,dataset=dataset,cluster_size=args.cluster_size)

 


if __name__ == '__main__':
 main()