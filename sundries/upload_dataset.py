import datasets
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Embedding Generation Script')


    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--hf_repo_name', type=str, required=True)
    

    return parser.parse_args()

if __name__ =='__main__':
    args= parse_args()
    dataset = datasets.load_from_disk(args.dataset_path)
    dataset.push_to_hub(args.hf_repo_name)