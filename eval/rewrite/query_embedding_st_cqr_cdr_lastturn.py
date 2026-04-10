"""
,,.
"""
import asyncio
import os
import torch
from torch.cuda import device_count
import numpy as np
import datasets
from loguru import logger
from transformers import AutoTokenizer,AutoModel
from torch.utils.data import DataLoader



def get_embedding(query_encoder,tokenizer,dataset):

 text = dataset['text']

 dataloader = DataLoader(text,batch_size=1024)
 query_embs=[]
 with torch.inference_mode():
 for batch in dataloader:


 query_ids = tokenizer(batch, max_length=query_encoder.config.max_position_embeddings, truncation=True,return_tensors="pt",padding=True)
 q_emb = query_encoder(input_ids=query_ids.input_ids, attention_mask=query_ids.attention_mask)
 q_emb = q_emb.last_hidden_state[:, 0,:]
 query_embs.append(q_emb)
 

 query_embs = torch.cat(query_embs, dim=0).cpu().numpy() # (num_query, hidden_dim)
 
 
 
 return query_embs

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
from shared.data_utils import parse_dataset



import argparse

def parse_args():
 parser = argparse.ArgumentParser(description='Embedding Generation Script')

 parser.add_argument('--embedding_output_path', type=str, required=True)
 parser.add_argument('--embedding_model_path', type=str, required=True)
 parser.add_argument('--embedding_model_name', type=str, required=True)
 parser.add_argument('--dataset_path', type=str, required=True)
 parser.add_argument('--dataset_name', type=str, required=True)

 

 return parser.parse_args()


# ----------------------------------------------------------------------------------------------------------
# The implementation here is migrated from the Neurips24 ChatQA repository: 
# https://huggingface.co/nvidia/dragon-multiturn-query-encoder/blob/evaluation/dataset.py

def formate_with_topic(instance):
 
 messages=instance['messages']

 text =""


 for message in messages[-3:]:
 if message['role'] == 'user':
 text+= f"User: this is a question about {instance['topic']}. {message['content']}\n" 
 else:
 text+= f"Agent: {message['content']}\n" 

 return {'text':text.strip()}



def formate(instance):
 
 messages=instance['messages']

 return {'text':messages[-2]['content'].strip()}




if __name__=='__main__':

 args = parse_args()
 

 embedding_model_path_list = args.embedding_model_path.split(',')
 embedding_model_name_list = args.embedding_model_name.split(',')
 dataset_path_list = args.dataset_path.split(',')
 dataset_name_list = args.dataset_name.split(',')

 
 for model_path,model_name in zip(embedding_model_path_list,embedding_model_name_list):

 engine_count = device_count() # device_count() 
 tokenizer = AutoTokenizer.from_pretrained(model_path)
 query_encoder = AutoModel.from_pretrained(model_path,trust_remote_code=True,device_map='auto')
 query_encoder.eval()
 for dataset_path,dataset_name in zip(dataset_path_list,dataset_name_list):

 embedding_output_dir = os.path.join(args.embedding_output_path,dataset_name)
 os.makedirs(embedding_output_dir, exist_ok=True) # Ensure directory exists for file checking
 output_file = os.path.join(embedding_output_dir, f"query_{model_name}_lastturn.npy")
 dataset = parse_dataset(dataset_path)


 # if os.path.exists(output_file):
 # logger.success(f"Found existing embedding file: {output_file}")
 if False:
...
 else:
 try:
 try:
 dataset = dataset['dev']
 except:
 dataset = dataset['test']
 except:
...
 

 dataset = dataset.map(formate)

 print(dataset[4]['text'])
 
 embeddings = get_embedding(query_encoder,tokenizer,dataset)

 
 # 4. Save newly computed embeddings
 logger.info(f"Saving embeddings to: {output_file}")
 
 np.save(output_file, embeddings)


 del query_encoder
 torch.cuda.empty_cache()