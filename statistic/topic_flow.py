from collections import Counter # Counter, 
import datasets
from tqdm import tqdm
import argparse

def parse_args():
 parser = argparse.ArgumentParser()
 parser.add_argument('--mtr-test-path', help="Path to the dataset disk location") # help 
 return parser.parse_args()

if __name__ == '__main__':
 args = parse_args()

 
 dataset = datasets.load_from_disk(args.mtr_test_path)


 total_array_transitions = [] # 

 for i in tqdm(range(0, len(dataset) - 7, 8), desc="Processing chunks"):

 id_mapping = {} # original_local_did new_id ( ID) 
 
 # 1. new_id = 1
 first_doc_original_idx = dataset[i]['ground_truth_ctx']['document_idx']
 id_mapping[first_doc_original_idx] = 1
 
 previous_remapped_id = 1 # ID
 next_available_remapped_id = 2 # ID

 current_chunk_transitions = []

 # ( 7, 7)
 for j in range(i, i + 8):
 if j >= len(dataset):
 break
 
 current_doc_original_idx = dataset[j]['ground_truth_ctx']['document_idx']
 current_doc_remapped_id = -1 # 

 if current_doc_original_idx in id_mapping:
 # document_idx, remapped_id
 current_doc_remapped_id = id_mapping[current_doc_original_idx]
 else:
 # document_idx (), next_available_remapped_id
 id_mapping[current_doc_original_idx] = next_available_remapped_id
 current_doc_remapped_id = next_available_remapped_id
 next_available_remapped_id += 1
 
 # remapped_id remapped_id 
 current_chunk_transitions.append((previous_remapped_id, current_doc_remapped_id))
 
 # previous_remapped_id 
 previous_remapped_id = current_doc_remapped_id
 
 if current_chunk_transitions: # 
 total_array_transitions.append(current_chunk_transitions)

 print(len(total_array_transitions))

 #, zip 
 num_positions = len(total_array_transitions[0])

 # Counter 
 positional_counters = []

 # 1: zip(*...) " " 
 # zip(*total_array_transitions):
 # [ [t1_pos0, t1_pos1,...], [t2_pos0, t2_pos1,...],...]
 # [ (t1_pos0, t2_pos0,...), (t1_pos1, t2_pos1,...),...]



 
 # transitions_by_position, 
 #, transitions_by_position[0] chunk_transitions (int,int) 
 transitions_by_position = list(zip(*total_array_transitions))

 for i, pos_transitions in enumerate(transitions_by_position):
 # pos_transitions, ((0,0), (0,0), (0,1), (0,0)) for position 0
 # Counter 
 position_counts = Counter(pos_transitions)
 positional_counters.append(position_counts)

 print(f"\nTransition Counts for Position {i}:")
 if not position_counts:
 print(" No transitions recorded for this position.")
 continue
 for transition, count in position_counts.most_common():
 print(f" {transition}: {count}")

