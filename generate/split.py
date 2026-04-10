import datasets
import os
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset_path')
    parser.add_argument('--output_dir')
    return parser.parse_args()



import math

def generate_spread_segments(total_length, num_segments):
    """
    Generate multiple spread-out contiguous index segments within range [0, total_length).

    Args:
        total_length (int): Upper bound of index range (exclusive).
        num_segments (int): Number of segments to generate.

    Returns:
        list[int]: A flat list of indices. Returns empty list if generation is not possible.
    """
    segment_len = 8

    if num_segments <= 0 or total_length < segment_len:
        return []

    # Calculate the last possible segment start position
    # A segment from s occupies s to s + segment_len - 1
    # So s + segment_len - 1 < total_length => s <= total_length - segment_len
    # s must also be a multiple of segment_len (8)
    max_valid_start_value = total_length - segment_len

    if max_valid_start_value < 0: # even starting from 0 cannot fit one segment
        return []

    # Max valid start "block" index (0=start at 0, 1=start at 8, etc.)
    # e.g.: total_length=16 => max_block_idx=1; total_length=8 => max_block_idx=0
    max_block_idx = max_valid_start_value // segment_len

    segments = []

    if num_segments == 1:
        # If only one segment, always start at position 0
        start_index = 0 
        segments.extend(list(range(start_index, start_index + segment_len)))
        return segments

    # For num_segments > 1, spread points evenly across [0, max_block_idx]
    # Similar to np.linspace(0, max_block_idx, num_segments)
    for i in range(num_segments):
        # Calculate ideal fractional block index
        # When i=0: 0; When i=num_segments-1: max_block_idx
        if num_segments - 1 == 0: # theoretically unreachable since num_segments=1 is handled
            ideal_fractional_block = 0
        else:
            ideal_fractional_block = i * max_block_idx / (num_segments - 1)
        
        # Round to nearest integer block index (traditional rounding for positive numbers)
        chosen_block_idx = int(ideal_fractional_block + 0.5)
        
        start_index = chosen_block_idx * segment_len
        
        # Ensure the segment doesn't exceed bounds (should be guaranteed by max_block_idx calculation)
        if start_index + segment_len <= total_length:
            segments.extend(list(range(start_index, start_index + segment_len)))
        # else:
            # Should not happen given the max_block_idx calculation.

    return segments



if __name__ == '__main__':

    
    args = parse_args()

    test_turn=1000
    dataset = datasets.load_from_disk(args.dataset_path)
    test_index = generate_spread_segments(len(dataset),test_turn)
    all_indices_set = set(range(len(dataset)))
    test_indices_set = set(test_index) # Convert test_index to a set for efficient difference

    train_indices_set = all_indices_set - test_indices_set
    train_index = sorted(list(train_indices_set)) # Convert back to a sorted list

    # dataset = dataset.train_test_split(test_size=0.1)
    train_dataset,test_dataset = dataset.select(train_index),dataset.select(test_index)
    train_dataset.save_to_disk(os.path.join(args.output_dir,'mtr_train'))
    test_dataset.save_to_disk(os.path.join(args.output_dir,'mtr_test'))