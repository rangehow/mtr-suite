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

os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

def initialize(model_name, model_path):
    return initialize_llm(model_name, model_path, args.inference_backend, context_length=60000)


if __name__=='__main__':

 args = parse_args()

 avaliable_dataset = ['qrecc'] # 'doc2dial','inscit',,'quac','topiocqa','coral'
 # avaliable_dataset = ['new_coral']
 if args.inference_backend == 'sglang':
 import sglang as sgl
 elif args.inference_backend == 'vllm':
 from vllm import LLM, SamplingParams
 else:
 logger.error(f"please set correcnt backend in ['sglang','vllm'], now {args.inference_backend}")
 assert False

 if args.inference_backend == 'sglang':
 assert False,"Not yet supported. Do not use this backend. If needed, implement sglang structured output yourself."
 logger.info("Run sglang backend")
 sampling_params = {"temperature": 0,'max_new_tokens':1024}
 elif args.inference_backend == 'vllm':
 # logger.info("Run vllm backend")
 # BUG 
 logger.warning("Workaround: Command-A model outputs <|END_RESPONSE|><EOS>, so we force stop_token_ids=[255022]. This may affect other models with ~250k vocabulary size (unlikely).")

 # from vllm.sampling_params import GuidedDecodingParams
 # from pydantic import BaseModel, Field
 # from enum import Enum
 # class JustificationRatingOutput(BaseModel):
 # Justification: str = Field(description="The explanation or reasoning.")
 # # You can make Rating more specific, e.g., int, float, or an Enum
 # Rating: str = Field(description="The overall rating (1-5).")
 # # If using the Enum:
 # # Rating: RatingValue = Field(description="The overall rating.")

 # # Get the JSON schema from the model
 # json_schema_justification = JustificationRatingOutput.model_json_schema()

 # # Create guided decoding parameters using the JSON schema
 # guided_decoding_params_json_justification = GuidedDecodingParams(json=json_schema_justification)

 # sampling_params = SamplingParams(temperature=0,max_tokens=1024,stop_token_ids=[255022],guided_decoding=guided_decoding_params_json_justification)

 sampling_params = SamplingParams(temperature=0,max_tokens=1024,stop_token_ids=[255022])

 tokenizer = AutoTokenizer.from_pretrained(args.model_path)
 
 


 for dataset_name in avaliable_dataset:
 output_dir = os.path.join(args.output_dir,f'{args.judge_model_name}_{dataset_name}')
 if dataset_name =='coral' or dataset_name=='new_coral':
 dataset = datasets.load_from_disk(args.coral_dir)
 else:
 dataset = datasets.load_dataset(args.chatrag_bench_dir,dataset_name)
 logger.info(dataset_name)

 if 'test' in dataset:
 dataset = dataset['test']
 else:
 dataset = dataset['dev']

 dataset = dataset.map(mock_chatrag,load_from_cache_file=False)

 # partial detect( only 1st turns)
 # idx=[]
 # for i in range(len(dataset)):
 # if len(dataset[i]['messages'])==2:
 # idx.append(i)
 # dataset = dataset.select(idx)
 

 if args.target == 'query':
 scoring_config = {
 'find_document':('tag_score',find_document),
 'judge_answerable': ('answerable_score', judge_answerable),
 
 # 'judge_atomic':('atomic_score',judge_atomic),
 # 'judge_explicit_reference':('reference_score',judge_explicit_reference),
 }
 elif args.target == 'response':

 scoring_config = {
 'judge_response_faithful':('faithful_score',judge_response_faithful),
 'judge_respnse_quality':('quality_score',judge_response_quality),
 }

 try:
 record_dataset = datasets.load_from_disk(output_dir)
 except Exception as e:
 record_dataset = dataset

 missing_scores_found = False
 for function_name, (score_col_name, scoring_func) in scoring_config.items():
 if score_col_name not in record_dataset.column_names:
 missing_scores_found = True
 logger.info(f"Score column '{score_col_name}' not found. Calculation needed.")
 
 if f'{score_col_name}_reason' not in record_dataset.column_names:

 try:
 if llm is None:
 logger.info("Initializing LLM for scoring...")
 llm = initialize(args.judge_model_name,args.model_path)
 except:
 llm = initialize(args.judge_model_name,args.model_path)
 else:
 logger.debug(f"Score column '{score_col_name}' already exists. Skipping llm initialization.")
 logger.info(f"Calling function '{function_name}' (mapped to {scoring_func.__name__}) to calculate '{score_col_name}'...")
 
 # dirty code just for debug usage
 try:
 print(llm)
 record_dataset = scoring_func(dataset, llm, sampling_params, tokenizer, args, record_dataset, score_col_name)
 except:
 record_dataset = scoring_func(dataset, None, sampling_params, tokenizer, args, record_dataset, score_col_name)
 logger.info(f"Finished calculating '{score_col_name}'.")



 if missing_scores_found:
 os.makedirs(output_dir,exist_ok=True)
 print(f"Missing scores detected, preparing to update dataset at: {output_dir}")

 temp_dir = None # Initialize temp dir variable
 move_successful = False # Flag whether move succeeded, for cleanup decision

 try:
 
 parent_dir = os.path.dirname(output_dir) or '.' # Get parent dir, if output_dir is top-level use'.'
 temp_dir_name = os.path.basename(output_dir) + f"_temp_{int(time.time())}"
 temp_dir = os.path.join(parent_dir, temp_dir_name)

 # Ensure temp dir does not exist, just in case (mkdtemp Usually guarantees uniqueness, but custom names need checking)
 if os.path.exists(temp_dir):
 shutil.rmtree(temp_dir) # If confirmed leftover from previous failure, can delete

 # os.makedirs(temp_dir) # If using custom name, need to create directory
 # Better to use tempfile.mkdtemp which creates dir atomically
 temp_dir = tempfile.mkdtemp(suffix="_dataset_update", prefix=os.path.basename(output_dir) + "_", dir=parent_dir)

 # 2. Save new dataset to temp directory
 try:
 
 record_dataset.save_to_disk(temp_dir)

 except Exception as e:
 print(f":Save dataset to temp dir '{temp_dir}': {e}")
 # If save fails, raise exception, terminate, and clean up temp dir in finally
 raise RuntimeError(f"Failed to save to temp dir: {e}") from e

 # 3. (Critical part of atomic operation) Check and delete old dataset directory
 if os.path.exists(output_dir):
 # Ensure target path is a directory (prevent accidental file deletion)
 if os.path.isdir(output_dir):
 
 try:
 shutil.rmtree(output_dir)
 except OSError as e:
 print(f":Unable to delete old directory '{output_dir}'. Reason: {e}")
 # Deletion failure is critical, new data cannot be moved in
 raise RuntimeError(f"Failed to delete old directory: {e}") from e
 else:
 # If output_dir exists but is not a directory, this is a problem
 raise RuntimeError(f":Target path '{output_dir}' exists but is not a directory.")

 # 4. Move temp directory (rename) to final location
 # shutil.move Usually atomic on same filesystem (fast rename)
 try:
 
 shutil.move(temp_dir, output_dir)
 move_successful = True # Mark move as successful
 except Exception as e:
 print(f":Unable to move temp directory '{temp_dir}' to '{output_dir}'. Reason: {e}")
 print(f"Important: New dataset may still remain in temp directory '{temp_dir}'.")
 # If move fails, also terminate
 raise RuntimeError(f"Failed to move temp directory: {e}") from e

 except Exception as e:
 # Catch any exception from save, delete, or move
 print(f"Serious error during dataset update, operation interrupted: {e}")
 # Can add more detailed error reporting or logging here

 finally:
 # 5. Cleanup: delete temp dir if still exists and move was unsuccessful
 # Check if temp_dir was created and still exists
 if temp_dir and os.path.exists(temp_dir) and not move_successful:
 print(f"Cleaning up temp directory left by failed operation: {temp_dir}")
 try:
 shutil.rmtree(temp_dir)
 print(f"Successfully cleaned temp directory: {temp_dir}")
 except OSError as e:
 # Cleanup failure should be logged as it may occupy disk space
 print(f":clean up temp directory '{temp_dir}' error: {e}")
 # elif move_successful:
 # print("Move succeeded, temp dir is now final dir, no cleanup needed.")


 else:
 print("No missing scores detected, skipping dataset update.")
 


 