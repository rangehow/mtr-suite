from collections import defaultdict
import datasets
import os
import argparse
from loguru import logger
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir')
    parser.add_argument('--output_dir')
    return parser.parse_args()


if __name__=="__main__":

    args = parse_args()
    
    turns = sorted(os.listdir(args.dataset_dir))

    logger.info(f'currently detect {len(turns)} turns, from turn {turns[0]} to turn {turns[-1]}')

    turn1_folder = os.path.join(args.dataset_dir,'1')
    dataset_in_specific_turn = sorted(os.listdir(turn1_folder), key=lambda x: int(x.split('-')[0]))

    
    # ['cluster_idx', 'ctxs', 'prompt_tokens', 'completion_tokens', 'ground_truth_document_idx', 'ground_truth_ctx', 'messages']
    total_data=defaultdict(list)
    for dataset_split in dataset_in_specific_turn:
        temp_dataset_list = []

        for turn in range(1,5):
            temp_dataset_dir = os.path.join(args.dataset_dir,str(turn),dataset_split)
            
            dataset = datasets.load_from_disk(temp_dataset_dir)
        
            temp_dataset_list.append(dataset)
        
        for i in range(len(temp_dataset_list[-1])):
            for dataset in temp_dataset_list:
                try:
                    total_data[(dataset_split,i)].append(dataset[i])
                except Exception as e:
                    print(e)


    # shuffle for hard switch

    import random

    # Extract all keys
    keys = list(total_data.keys())

    # Randomly shuffle keys
    random.shuffle(keys)

    # Combine into pairs
    pairs = []
    # First loop: build pairs data
    for i in range(0, len(keys), 2):
        if i + 1 >= len(keys):
            break  # Skip the last unpaired key if it exists
        key1 = keys[i]
        key2 = keys[i + 1]
        pair_data = (total_data[key1], total_data[key2])
        pairs.append(pair_data)

    final_data=[]
    # Second loop: merge messages
    for i, (first_data, second_data) in enumerate(pairs):

        need_to_focused = first_data[-1]['messages']

        # Prepend first_data's messages to each record in second_data
        for item in second_data:
            item['messages'] = need_to_focused + item['messages']


        # pairs[i] = (first_data, second_data)
        final_data.extend(first_data)
        final_data.extend(second_data)

    
    dataset = datasets.Dataset.from_list(final_data)

    dataset.save_to_disk(args.output_dir)  


            
