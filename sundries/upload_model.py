from transformers import AutoModel,AutoTokenizer
import argparse


def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--hf_repo_name', type=str, required=True)
    

    return parser.parse_args()

if __name__ =='__main__':
    args= parse_args()
    model = AutoModel.from_pretrained(args.model_path)
    model.push_to_hub(args.hf_repo_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.push_to_hub(args.hf_repo_name)