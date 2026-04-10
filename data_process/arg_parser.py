import argparse

def parse_args():
 """ """
 parser = argparse.ArgumentParser(description=' ')
 
 parser.add_argument('--dataset_path', 
 type=str, 
 required=True,
 help=' ( imdb./local/path)')
 
 parser.add_argument('--save_dir', 
 type=str, 
 required=True,
 help=' ( imdb./local/path)')
 
 parser.add_argument('--max_length', 
 type=int, 
 default=2048,
 help=' ( imdb./local/path)')
 
 parser.add_argument('--quality_model_path')
 parser.add_argument('--fineweb_model_path')
 
 return parser.parse_args()