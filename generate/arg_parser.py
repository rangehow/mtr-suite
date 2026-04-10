import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--output_dir')
    parser.add_argument('--dataset_path')
    parser.add_argument('--query_model_name')
    parser.add_argument('--query_model_path')
    parser.add_argument('--response_model_name')
    parser.add_argument('--response_model_path')

    parser.add_argument('--turn')
    
    parser.add_argument('--cache_dir')
    parser.add_argument('--inference_backend')

    parser.add_argument('--last_turn_dataset')
    return parser.parse_args()