from functools import partial

from transformers import AutoTokenizer
import numpy as np
import json
import datasets
import re
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from torch.cuda import device_count
from arg_parser import parse_args
from tqdm import tqdm
from map_func import *
from loguru import logger
from answerable import judge_answerable,find_document,judge_atomic,judge_explicit_reference,judge_response_faithful,judge_response_quality
import shutil
import tempfile
import time
from shared.llm_utils import initialize_llm

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
os.environ['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN']='1'

def initialize(model_name, model_path):
    return initialize_llm(model_name, model_path, args.inference_backend)

if __name__=="__main__":
 args=parse_args()
 tested_model_name = args.tested_model_name
 output_dir_base = args.output_dir
 turn = args.turn
 start=args.start
 end=args.end

 logger.debug(args)

 if args.inference_backend == 'sglang':
 import sglang as sgl
 elif args.inference_backend == 'vllm':
 from vllm import LLM, SamplingParams
 else:
 logger.error(f"please set correcnt backend in ['sglang','vllm'], now {args.inference_backend}")
 assert False

 tokenizer = AutoTokenizer.from_pretrained(args.model_path)

 if args.inference_backend == 'sglang':
 assert False,"Not yet supported. Do not use this backend. If needed, implement sglang structured output yourself."
 sampling_params = {"temperature": 0,'max_new_tokens':1024}
 elif args.inference_backend == 'vllm':
 logger.info("Run vllm backend")
 logger.warning("Workaround: Command-A model outputs <|END_RESPONSE|><EOS>, so we force stop_token_ids=[255022]. This may affect other models with ~250k vocabulary size (unlikely).")
 sampling_params = SamplingParams(temperature=0,max_tokens=1024,stop_token_ids=[255022])

 # --- Main modification start () ---

 tasks_config = {
 'judge_answerable': ('answerable_score', judge_answerable),
 'find_document': ('tag_score', find_document),
 'judge_response_faithful': ('faithful_score', judge_response_faithful),
 'judge_respnse_quality': ('quality_score', judge_response_quality),
 }

 # 
 output_dir = os.path.join(output_dir_base, f'{args.judge_model_name}-{args.tested_model_name}')
 
 source_data_path = os.path.join(args.input_dir, f'{tested_model_name}-{tested_model_name}', turn, f'{start}-{end}')

 # 1.:, 
 try:
 main_dataset = datasets.load_from_disk(output_dir)
 logger.info(f"Successfully loaded existing dataset from: {output_dir}")
 except Exception:
 logger.info(f"No existing dataset found at {output_dir}. Loading from source: {source_data_path}")
 if not os.path.exists(source_data_path):
 logger.error(f"Source data path does not exist: {source_data_path}. Aborting.")
 exit(1)
 main_dataset = datasets.load_from_disk(source_data_path)

 llm = None # LLM
 scores_were_calculated = False # 


 for function_name, (score_col_name, scoring_func) in tasks_config.items():
 score_exists = score_col_name in main_dataset.column_names
 reason_exists = f'{score_col_name}_reason' in main_dataset.column_names

 # tag_score 
 if not score_exists or not reason_exists or score_col_name == 'tag_score':
 scores_were_calculated = True
 logger.info(f"Missing '{score_col_name}' or its reason. Calculation needed.")

 # LLM
 if llm is None:
 logger.info("Initializing LLM for scoring...")
 llm = initialize(args.judge_model_name, args.model_path)
 
 logger.info(f"Calling function '{function_name}' to calculate '{score_col_name}'...")
 
 # ** **: main_dataset
 #: `scoring_func` `main_dataset` `main_dataset`
 #: def judge_answerable(dataset, llm,...): 
 # scores =... # calculate scores from dataset
 # return dataset.add_column('answerable_score', scores)
 # scoring_func(dataset, llm,..., record_dataset,...), 
 #, scoring_func(dataset_to_process, llm,..., score_col_name)
 # 
 main_dataset = scoring_func(main_dataset, llm, sampling_params, tokenizer, args, score_col_name)

 logger.info(f"Finished calculating '{score_col_name}'.")
 else:
 logger.debug(f"Score column '{score_col_name}' already exists. Skipping.")


 if scores_were_calculated:
 logger.info(f"Scores have been updated. Saving the final dataset to {output_dir}")
 os.makedirs(output_dir, exist_ok=True)
 temp_dir = tempfile.mkdtemp(dir=os.path.dirname(output_dir) or '.')
 try:
 main_dataset.save_to_disk(temp_dir)
 if os.path.exists(output_dir):
 shutil.rmtree(output_dir)
 shutil.move(temp_dir, output_dir)
 logger.info(f"Dataset successfully saved to {output_dir}")
 except Exception as e:
 logger.error(f"Failed to save dataset: {e}")
 if os.path.exists(temp_dir):
 shutil.rmtree(temp_dir) # 
 # finally 
 else:
 logger.info("All scores already exist. No updates were made.")