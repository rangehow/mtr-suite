import os
import numpy as np
import torch
from vllm import LLM


def get_embedding(model_name, dataset, model_name_or_path, embedding_output_dir):
 """
 vLLM Embedding.

 Args:
 model_name (str): ( save file)
 dataset (dict): 'text', 
 model_name_or_path (str): Hugging Face 
 embedding_output_dir (str): 

 Returns:
 numpy.ndarray: 
 """
 os.makedirs(embedding_output_dir, exist_ok=True)
 output_file = os.path.join(embedding_output_dir, f"{model_name}.npy")

 # 
 if os.path.exists(output_file):
 print(f": {output_file}")
 try:
 embeddings = np.load(output_file, allow_pickle=True)
 print(".")
 return embeddings
 except Exception as e:
 print(f",Will recompute: {e}")

 print(f" vLLM: {model_name_or_path}")
 model = LLM(model=model_name_or_path, task="embed")

 texts = dataset['text']
 if not texts:
 print(": dataset['text'].")
 return np.array([])

 print(f" {len(texts)},Starting embedding computation...")
 outputs = model.embed(texts)
 embeddings = torch.tensor([o.outputs.embedding for o in outputs]).numpy()

 print(f",: {embeddings.shape}")

 # 
 np.save(output_file, embeddings)
 print(f": {output_file}")

 return embeddings
