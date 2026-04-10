import argparse



def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',type=int)
    parser.add_argument('--end',type=int)
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--model_path')
    parser.add_argument('--turn')
    parser.add_argument('--inference_backend')
    parser.add_argument('--tested_model_name')
    parser.add_argument('--judge_model_name')
    return parser.parse_args()