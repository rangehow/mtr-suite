import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--chatrag-bench-dir')
    parser.add_argument('--coral-dir')
    parser.add_argument('--target',help='judge query or response?')
    parser.add_argument('--output_dir')
    parser.add_argument('--model_path')
    parser.add_argument('--turn')
    parser.add_argument('--inference_backend')
    parser.add_argument('--judge_model_name')
    return parser.parse_args()