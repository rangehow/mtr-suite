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
random.seed(0)


# def jaccard_similarity(set1, set2):
# intersection = set1.intersection(set2)
# union = set1.union(set2)
# return len(intersection) / len(union)


# def generate_ngrams(text, n=5):
# for i in range(len(text) - n + 1):
# ngram = text[i: i + n]
# ngrams.add(ngram)
# return ngrams


# def ngram_similarity(doc1, doc2, n):
# ngrams_doc1 = generate_ngrams(doc1, n)
# ngrams_doc2 = generate_ngrams(doc2, n)
# return jaccard_similarity(ngrams_doc1, ngrams_doc2)



def _process_cluster_chunk(args):
 """
 cluster_idx chunk.

 Args:
 args (tuple): chunk_of_indices (list), dataset,
 all_cluster_docs ( cluster2docs).

 Returns:
 list: chunk.
 chunk,.
 """
 chunk_indices, dataset, all_cluster_docs = args
 results_for_chunk = []
 try:
 # cluster_idx
 for cluster_idx in chunk_indices:
 # all_cluster_docs 
 cluster_doc_indices = all_cluster_docs[cluster_idx]
 # 
 cluster_data = dataset.select(cluster_doc_indices).to_list()
 
 if len(cluster_data)==0:
 logger.error(f' idx ctx 0,cluster_idx{cluster_idx},cluster_doc_indices{cluster_doc_indices}')
 

 for i, doc_dict in enumerate(cluster_data):
 doc_index = cluster_doc_indices[i]
 doc_dict['document_idx'] = doc_index
 results_for_chunk.append({'cluster_idx': cluster_idx, 'ctxs': cluster_data})
 return results_for_chunk # chunk 
 except Exception as e:
 #, chunk 
 print(f"Error processing chunk starting with index {chunk_indices[0] if chunk_indices else 'N/A'}: {e}")
 #, chunk 
 return []



class sort_class:
 def __init__(self, cluster_size, dataset, neighbors,cluster_dataset_output_dir):
 self.dataset = dataset
 self.knns = neighbors
 self.num_docs = len(dataset)
 self.seen_docs = set()
 self.unseen_docs = set(range(self.num_docs))
 print(f"num docs: {self.num_docs}")
 self.cluster_dataset_output_dir=cluster_dataset_output_dir
 # 
 self.doc_sim_threshold = 0.85
 self.n = 3 # n-gram


 self.cluster_size = cluster_size
 self.cur_k = None
 self.filter_docs = []
 self.cluster2docs = defaultdict(list)
 self.doc2cluster = {}



 def check_cluster_sizes(self):
 oversized_clusters = []
 for cluster_id, docs in self.cluster2docs.items():
 if len(docs) > self.cluster_size:
 oversized_clusters.append(cluster_id)

 if oversized_clusters:
 print(f" {self.cluster_size}:")
 for cluster_id in oversized_clusters:
 print(f" {cluster_id}: {len(self.cluster2docs[cluster_id])} ")
 else:
 print(f" {self.cluster_size}")

 def sort(self):

 cluster_id = 0
 #, 1 chunk cluster_size
 cur_cluster_len = 1

 self.cur_k = self.unseen_docs.pop()
 self.cluster2docs[cluster_id].append(self.cur_k)
 self.doc2cluster[self.cur_k] = cluster_id
 self.seen_docs.add(self.cur_k)

 with tqdm(total=self.num_docs - 1,desc='sort') as pbar:
 while self.unseen_docs:
 knn = self.knns[self.cur_k,:]
 first_doc = self.output_first_doc_knn(knn)
 if (first_doc is None) or (cur_cluster_len >= self.cluster_size):

 self.cur_k = self.unseen_docs.pop()
 cluster_id += 1
 cur_cluster_len = 0
 else:
 self.cur_k = first_doc
 self.unseen_docs.remove(self.cur_k)

 self.cluster2docs[cluster_id].append(self.cur_k)
 self.doc2cluster[self.cur_k] = cluster_id
 cur_cluster_len += 1
 self.seen_docs.add(self.cur_k)
 pbar.update(1)
 print(":", len(self.cluster2docs))
 self.analysis()

 def output_first_doc_knn_not_in_the_cluster(self, knn, cluster_id):
 for k in knn:
 if k!= -1:
 k_cluster = self.doc2cluster[k]

 while (
 k_cluster!= cluster_id
):
 return k, k_cluster

 return None, None

 def check_all_docs_assigned(self, cluster2docs):
 all_docs = set()
 for cluster in cluster2docs.values():
 all_docs.update(cluster)

 all_docs = sorted(all_docs)

 
 print(f" {len(all_docs)}")

 for i, doc in enumerate(all_docs):
 if i!= doc:
 print(f": {i} {doc - 1}")
 return False

 print(" ")
 return True


 def analysis(self):
 print("\n--- ---")

 cluster_sizes = [len(docs) for docs in self.cluster2docs.values()]

 if not cluster_sizes:
 print(".")
 return # 

 cluster_sizes_np = np.array(cluster_sizes)

 min_size = np.min(cluster_sizes_np)
 max_size = np.max(cluster_sizes_np)
 mean_size = np.mean(cluster_sizes_np)
 median_size = np.median(cluster_sizes_np) # (50)

 # Percentiles (25%, 75%, 90%, 95%)
 percentile_25 = np.percentile(cluster_sizes_np, 25)
 percentile_75 = np.percentile(cluster_sizes_np, 75)
 percentile_90 = np.percentile(cluster_sizes_np, 90)
 percentile_95 = np.percentile(cluster_sizes_np, 95)

 print(f": {len(self.cluster2docs)}")
 print(f": {min_size}")
 print(f": {max_size}")
 print(f": {mean_size:.2f}") # 
 print(f" (50): {median_size}")
 print(f" 25: {percentile_25}")
 print(f" 75: {percentile_75}")
 print(f" 90: {percentile_90}")
 print(f" 95: {percentile_95}")
 print("-----------------------")

 def merge(self):
 # self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs.pk")
 # self.doc2cluster = pickle_load(f"{self.output_file}/doc2cluster.pk")

 merged_clusters_num = 0

 # self.cluster2docs copy
 for cluster in tqdm(list(self.cluster2docs.keys())):
 if cluster not in self.cluster2docs:
 continue

 cluster_docs = self.cluster2docs[cluster]

 if len(cluster_docs) < self.cluster_size:
 merged_clusters_num += 1
 # print(merged_clusters_num)
 for doc in cluster_docs:
 knn = self.knns[doc,:]

 top1k, top1k_cluster = self.output_first_doc_knn_not_in_the_cluster(
 knn, cluster
)

 if top1k_cluster in self.cluster2docs:
 k_cluster_docs = self.cluster2docs[top1k_cluster]

 # top1k 
 if top1k in k_cluster_docs:
 k_cluster_docs.insert(k_cluster_docs.index(top1k), doc)

 # update the cluster
 self.cluster2docs[top1k_cluster] = k_cluster_docs
 self.doc2cluster[doc] = top1k_cluster
 else:
 # top1k, fallback, 
 self.cluster2docs[top1k_cluster].append(doc)
 self.doc2cluster[doc] = top1k_cluster
 else:
 print(f"Warning: Target cluster {top1k_cluster} not found for doc {doc}. Skipping doc.")
 # 
 # pass # Or handle the unassigned doc

 #, entry
 del self.cluster2docs[cluster]

 print(
 f"merged_clusters_num:{merged_clusters_num},: {len(self.cluster2docs)}"
)

 print("Starting cluster re-indexing...")
 new_cluster2docs = {}
 new_doc2cluster = {}
 current_new_id = 0

 # ID, (, ID ID)
 #, ID 
 remaining_old_cluster_ids = sorted(self.cluster2docs.keys())

 for old_cluster_id in tqdm(remaining_old_cluster_ids, desc="Re-indexing clusters"):
 docs_in_cluster = self.cluster2docs[old_cluster_id]
 new_cluster2docs[current_new_id] = docs_in_cluster

 # doc2cluster ID
 for doc in docs_in_cluster:
 new_doc2cluster[doc] = current_new_id

 current_new_id += 1

 # 
 self.cluster2docs = new_cluster2docs
 self.doc2cluster = new_doc2cluster

 print(f"Re-indexing complete. Final number of clusters: {len(self.cluster2docs)}")

 # 
 self.analysis()

 self.check_all_docs_assigned(self.cluster2docs)
 

 def output_first_doc_knn(self, knn):

 for k in knn:
 if k not in self.seen_docs and k!= -1:
 return k
 return None

 
 # def write_docs(self):
 

 # new_dataset_list=[]
 # for cluster_idx in tqdm(range(len(self.cluster2docs))):
 # cluster_data=self.dataset.select(self.cluster2docs[cluster_idx]).to_list()
 # new_dataset_list.append({'cluster_idx':cluster_idx,'data':cluster_data})
 
 # new_dataset = datasets.Dataset.from_list(new_dataset_list)
 # new_dataset.save_to_disk(self.cluster_dataset_output_dir,num_proc=64)

 def write_docs(self, num_data_prep_processes=None, num_save_processes=64):
 """
 (),.

 Args:
 num_data_prep_processes (int, optional):.
 CPU.
 num_save_processes (int, optional): save_to_disk.
 64.
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
 # Dataset 
 empty_dataset = datasets.Dataset.from_dict({}) # schema 
 print(f"Saving empty dataset structure to {self.cluster_dataset_output_dir}...")
 empty_dataset.save_to_disk(self.cluster_dataset_output_dir, num_proc=1)
 print("Finished.")
 return

 # 
 num_data_prep_processes = min(num_data_prep_processes, num_clusters)
 if num_data_prep_processes <= 0:
 num_data_prep_processes = 1 # 

 # 1. cluster_idx num_data_prep_processes 
 all_indices = list(range(num_clusters))
 chunks = [all_indices[i::num_data_prep_processes] for i in range(num_data_prep_processes)]
 chunks = [chunk for chunk in chunks if chunk]

 print(f"Divided {num_clusters} clusters into {len(chunks)} chunks for processing.")


 # (chunk) 
 #: self.cluster2docs 
 tasks = [(chunk, self.dataset, self.cluster2docs) for chunk in chunks]

 results_list_of_lists = []

 print("Starting parallel data preparation (chunked)...")
 with multiprocessing.Pool(processes=num_data_prep_processes) as pool:
 # imap_unordered, 
 # tqdm chunk 
 results_iterator = pool.imap_unordered(_process_cluster_chunk, tasks)
 for chunk_result in tqdm(results_iterator, total=len(tasks), desc="Processing chunks in parallel"):
 # 4. Collect results from each chunk
 if chunk_result:
 results_list_of_lists.append(chunk_result)

 print("Parallel data preparation finished.")

 # 5. (Flatten the list of lists)
 print("Merging results from chunks...")
 final_results_unordered = [item for sublist in results_list_of_lists for item in sublist]


 print("Sorting final results...")
 final_results_sorted = sorted(final_results_unordered, key=lambda x: x['cluster_idx'])

 # 7. Dataset 
 print("Creating final dataset from prepared data...")
 if not final_results_sorted:
 print("Warning: No data was successfully processed.")
 new_dataset = datasets.Dataset.from_dict({}) # 
 else:
 new_dataset = datasets.Dataset.from_list(final_results_sorted)


 print(f"Saving dataset in parallel to {self.cluster_dataset_output_dir} using {num_save_processes} processes...")
 new_dataset.save_to_disk(self.cluster_dataset_output_dir, num_proc=num_save_processes)
 print("Chunked parallel version finished.")

def get_cluster_dataset(cluster_dataset_output_dir,neighbors,dataset,cluster_size):
 Path(cluster_dataset_output_dir).mkdir(parents=True, exist_ok=True)

 sort_member = sort_class(
 cluster_size=cluster_size,
 dataset=dataset, 
 neighbors=neighbors,
 cluster_dataset_output_dir=cluster_dataset_output_dir,
)

 sort_member.sort()
 sort_member.merge()
 sort_member.write_docs()

if __name__ == "__main__":
...
 

 
 
 