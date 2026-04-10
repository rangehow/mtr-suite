import argparse
import datasets
import os

def main():
 parser = argparse.ArgumentParser(description="Filter and sample dataset")
 parser.add_argument('--dataset_path', type=str, required=True, help="Path to the unfiltered dataset")
 parser.add_argument('--save_path', type=str, required=True, help="Path to save the filtered dataset")
 parser.add_argument('--sample_save_path', type=str, default=None, help="Optional path to save shuffled sample dataset")
 args = parser.parse_args()

 # 
 print(f"Loading dataset from {args.dataset_path}...")
 dataset = datasets.load_from_disk(args.dataset_path)

 # 
 print("Filtering dataset...")
 filtered_dataset = dataset.filter(lambda x: x['naive_quality'] == 0 and x['edu_quality'] > 1.9, num_proc=64)
 print(f"Filtered dataset size: {len(filtered_dataset)}")

 # 
 print(f"Saving filtered dataset to {args.save_path}...")
 filtered_dataset.save_to_disk(args.save_path)

 # sample_save_path, 
 if args.sample_save_path:
 print("Creating shuffled sample dataset to match filtered dataset size...")
 shuffled_dataset = dataset.shuffle()
 sampled_dataset = shuffled_dataset.select(range(len(filtered_dataset)))

 print(f"Saving sampled dataset ({len(sampled_dataset)} samples) to {args.sample_save_path}...")
 sampled_dataset.save_to_disk(args.sample_save_path)
 print("Sample dataset saved successfully.")

 print("All processing complete.")

if __name__ == '__main__':
 main()
