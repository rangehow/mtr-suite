import asyncio
import os
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine
import torch
from torch.cuda import device_count
import numpy as np
import datasets
from loguru import logger



import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
from shared.embedding_utils import split_sentences, embed_text, run_parallel_embeddings
def get_embedding(array, dataset):
 """
 Get or load text embeddings for the specified model.

 If embedding file exists, load directly. Otherwise, compute and save.

 Args:
 model_name (str): Name used to identify the model, also used for filename.
 dataset (dict): Dataset containing 'text' key with text list or array.
 model_name_or_path (str): Model name or path, passed to embedding engine.
 embedding_output_dir (str): Directory to save embedding files.

 Returns:
 numpy.ndarray: Computed or loaded embedding vector array.
 """


 texts_to_embed = dataset['text']
 if not texts_to_embed:
 logger.error("Input dataset's 'text' field is empty.")
 
 return np.array([])

 # Run parallel embedding computation
 print("Running embedding computation...")
 results = asyncio.run(run_parallel_embeddings(array, texts_to_embed, engine_count))
 print("Embedding computation complete.")

 print(f"Embedding results preview (shape): {results.shape if isinstance(results, np.ndarray) else 'not a Numpy array'}") # 
 
 
 
 return results

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
from shared.data_utils import parse_dataset_with_truncation as parse_dataset


import argparse

def parse_args():
 parser = argparse.ArgumentParser(description='Embedding Generation Script')

 parser.add_argument('--embedding_output_path', type=str, required=True)
 parser.add_argument('--embedding_model_path', type=str, required=True)
 parser.add_argument('--embedding_model_name', type=str, required=True)
 parser.add_argument('--dataset_path', type=str, required=True)
 parser.add_argument('--dataset_name', type=str, required=True)

 

 return parser.parse_args()


async def cleanup_resources():
 print("Stopping array asynchronously...")
 if hasattr(array, 'astop') and callable(array.astop):
 await array.astop() # Now 'await' is inside an 'async def' function
 else:
 print("Warning: 'array' object does not have a callable 'astop' method.")
 logger.info("Array stopped.")

if __name__=='__main__':

 args = parse_args()
 

 embedding_model_path_list = args.embedding_model_path.split(',')
 embedding_model_name_list = args.embedding_model_name.split(',')
 dataset_path_list = args.dataset_path.split(',')
 dataset_name_list = args.dataset_name.split(',')

 
 for model_path,model_name in zip(embedding_model_path_list,embedding_model_name_list):
 
 assert 'docchat' not in model_name.lower(),"infinity->sentence transformer use mean pooling which harms dragon+ like model huge!"

 engine_count = device_count() # device_count() 
 array = AsyncEngineArray.from_args([
 EngineArgs(
 batch_size= 16 if 'qwen' in model_name.lower() else 256,
 model_name_or_path=model_path,
 engine="torch",
 embedding_dtype="float32", 
 dtype="auto",
 device='cuda',
 device_id=f'{i}',
 model_warmup=False,
 served_model_name=f'model_{i}',
 lengths_via_tokenize=True,
 # compile=False if 'modernbert' in model_name.lower() else True
) for i in range(engine_count)
])

 for dataset_path,dataset_name in zip(dataset_path_list,dataset_name_list):
 
 embedding_output_dir = os.path.join(args.embedding_output_path,dataset_name)
 os.makedirs(embedding_output_dir, exist_ok=True) # Ensure directory exists for file checking
 output_file = os.path.join(embedding_output_dir, f"document_{model_name}.npy")
 

 # max_len = max([len(x) for x in dataset['text']])
 # from transformers import AutoTokenizer
 # tokenizer = AutoTokenizer.from_pretrained(model_path)
 # print(tokenizer(dataset['text'],padding=True,return_tensors='pt').input_ids.shape)
 # logger.info(f"max len:{max_len}")
 # if dataset_name=='qrecc':
 # max_len=-1
 # for i, d in enumerate(dataset):
 # current_len = len(d['text'])
 # if current_len > max_len:
 # max_len = current_len
 # max_index = i
 # longest_item = d # Store the dictionary itself

 if os.path.exists(output_file):
 logger.success(f"Found existing embedding file: {output_file}")
 
 else:
 dataset = parse_dataset(dataset_path,dataset_name)
 logger.info(dataset_name)
 embeddings = get_embedding(array,dataset)

 
 # 4. Save newly computed embeddings
 logger.info(f"Saving embeddings to: {output_file}")
 
 np.save(output_file, embeddings)
 

 asyncio.run(cleanup_resources())
 torch.cuda.empty_cache()