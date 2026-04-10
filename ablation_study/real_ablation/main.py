import gc
import sys
from tqdm import tqdm
import argparse
import pickle
from collections import OrderedDict, defaultdict
import statistics
from pathlib import Path
import random
from multiprocessing import Pool
import multiprocessing
import numpy as np
import time
import json
import os
from rich.progress import track
import datasets
from loguru import logger

# Set random seed for reproducibility
random.seed(0)

def _process_cluster_chunk(args):
 """
 cluster_idx chunk.
 """
 chunk_indices, dataset, all_cluster_docs = args
 results_for_chunk = []
 try:
 for cluster_idx in chunk_indices:
 cluster_doc_indices = all_cluster_docs[cluster_idx]
 
 if not cluster_doc_indices:
 logger.warning(f'Cluster_idx {cluster_idx} is empty.')
 continue

 cluster_data = dataset.select(cluster_doc_indices).to_list()
 
 if len(cluster_data) == 0:
 logger.error(f'Could not retrieve data for cluster_idx {cluster_idx} with indices {cluster_doc_indices}')
 continue

 for i, doc_dict in enumerate(cluster_data):
 doc_index = cluster_doc_indices[i]
 doc_dict['document_idx'] = doc_index
 results_for_chunk.append({'cluster_idx': cluster_idx, 'ctxs': cluster_data})
 return results_for_chunk
 except Exception as e:
 print(f"Error processing chunk starting with index {chunk_indices[0] if chunk_indices else 'N/A'}: {e}")
 return []


class RandomClusterer:
 """
.
 """
 def __init__(self, cluster_size, dataset, cluster_dataset_output_dir):
 self.dataset = dataset
 self.num_docs = len(dataset)
 self.cluster_size = cluster_size
 self.cluster_dataset_output_dir = cluster_dataset_output_dir

 self.cluster2docs = {}
 print(f"RandomClusterer initialized for {self.num_docs} documents with cluster size {self.cluster_size}.")

 def create_clusters(self):
 """
:.
 """
 print("Starting random clustering...")
 

 doc_indices = list(range(self.num_docs))
 

 print("Shuffling document indices...")
 random.shuffle(doc_indices)
 
 # 3. cluster_size 
 print(f"Chunking indices into clusters of size {self.cluster_size}...")
 cluster_id = 0
 for i in tqdm(range(0, self.num_docs, self.cluster_size), desc="Creating random clusters"):
 chunk = doc_indices[i: i + self.cluster_size]
 if chunk: # 
 self.cluster2docs[cluster_id] = chunk
 cluster_id += 1
 
 print(f"Random clustering finished. Created {len(self.cluster2docs)} clusters.")
 
 self.analysis()
 self.check_all_docs_assigned()

 def analysis(self):
 """."""
 print("\n--- Cluster Size Distribution Analysis ---")
 if not self.cluster2docs:
 print("No clusters to analyze.")
 return

 cluster_sizes = [len(docs) for docs in self.cluster2docs.values()]
 cluster_sizes_np = np.array(cluster_sizes)

 print(f"Total number of clusters: {len(self.cluster2docs)}")
 print(f"Min cluster size: {np.min(cluster_sizes_np)}")
 print(f"Max cluster size: {np.max(cluster_sizes_np)}")
 print(f"Mean cluster size: {np.mean(cluster_sizes_np):.2f}")
 print(f"Median cluster size: {np.median(cluster_sizes_np)}")
 print("------------------------------------------")

 def check_all_docs_assigned(self):
 """."""
 all_docs_in_clusters = set()
 for docs in self.cluster2docs.values():
 all_docs_in_clusters.update(docs)

 print(f"\nTotal unique documents assigned to clusters: {len(all_docs_in_clusters)}")
 print(f"Total documents in dataset: {self.num_docs}")

 if len(all_docs_in_clusters) == self.num_docs:
 print("Success: All documents have been assigned to a cluster exactly once.")
 return True
 else:
 print("Error: Mismatch in document count. Some documents might be missing or duplicated.")
 # 
 # missing_docs = set(range(self.num_docs)) - all_docs_in_clusters
 # print(f"Number of missing documents: {len(missing_docs)}")
 return False

 # --- write_docs, ---
 def write_docs(self, num_data_prep_processes=None, num_save_processes=64):
 """
.
,.
 """
 if num_data_prep_processes is None:
 try:
 num_data_prep_processes = multiprocessing.cpu_count()
 except NotImplementedError:
 num_data_prep_processes = 4
 print(f"Using {num_data_prep_processes} processes for data preparation (defaulting to CPU count).")
 else:
 print(f"Using {num_data_prep_processes} processes for data preparation.")

 num_clusters = len(self.cluster2docs)

 if num_clusters == 0:
 print("No clusters to process.")
 empty_dataset = datasets.Dataset.from_dict({})
 print(f"Saving empty dataset structure to {self.cluster_dataset_output_dir}...")
 empty_dataset.save_to_disk(self.cluster_dataset_output_dir, num_proc=1)
 print("Finished.")
 return

 num_data_prep_processes = min(num_data_prep_processes, num_clusters)
 if num_data_prep_processes <= 0:
 num_data_prep_processes = 1

 all_indices = list(self.cluster2docs.keys())
 chunks = [all_indices[i::num_data_prep_processes] for i in range(num_data_prep_processes)]
 chunks = [chunk for chunk in chunks if chunk]

 print(f"Divided {num_clusters} clusters into {len(chunks)} chunks for processing.")
 
 tasks = [(chunk, self.dataset, self.cluster2docs) for chunk in chunks]

 results_list_of_lists = []
 print("Starting parallel data preparation...")
 with multiprocessing.Pool(processes=num_data_prep_processes) as pool:
 results_iterator = pool.imap_unordered(_process_cluster_chunk, tasks)
 for chunk_result in tqdm(results_iterator, total=len(tasks), desc="Processing chunks in parallel"):
 if chunk_result:
 results_list_of_lists.append(chunk_result)

 print("Parallel data preparation finished.")
 print("Merging results from chunks...")
 final_results_unordered = [item for sublist in results_list_of_lists for item in sublist]

 print("Sorting final results by cluster_idx...")
 final_results_sorted = sorted(final_results_unordered, key=lambda x: x['cluster_idx'])

 print("Creating final dataset from prepared data...")
 if not final_results_sorted:
 print("Warning: No data was successfully processed.")
 new_dataset = datasets.Dataset.from_dict({})
 else:
 new_dataset = datasets.Dataset.from_list(final_results_sorted)

 print(f"Saving dataset in parallel to {self.cluster_dataset_output_dir} using {num_save_processes} processes...")
 new_dataset.save_to_disk(self.cluster_dataset_output_dir, num_proc=num_save_processes)
 print("Dataset saved successfully.")


def get_cluster_dataset(cluster_dataset_output_dir, dataset, cluster_size):
 """
,.
: neighbors.
 """
 Path(cluster_dataset_output_dir).mkdir(parents=True, exist_ok=True)


 clusterer = RandomClusterer(
 cluster_size=cluster_size,
 dataset=dataset, 
 cluster_dataset_output_dir=cluster_dataset_output_dir,
)


 clusterer.create_clusters()
 

 clusterer.write_docs()

if __name__ == "__main__":
 # 


 dataset = datasets.load_from_disk('/path/to/mtr/mtr-data-dumps/processed_dataset')
 

 CLUSTER_SIZE = 8
 OUTPUT_DIR = "/path/to/mtr/mtr-data-dumps/random_clustered_dataset"
 
 print(f"Total documents: {len(dataset)}")
 print(f"Target cluster size: {CLUSTER_SIZE}")
 print(f"Output directory: {OUTPUT_DIR}")


 #, `neighbors` 
 get_cluster_dataset(
 cluster_dataset_output_dir=OUTPUT_DIR,
 dataset=dataset,
 cluster_size=CLUSTER_SIZE
)

 print("\nDemonstration finished. Check the output in the './random_clustered_dataset' directory.")