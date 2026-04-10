import asyncio
import os
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine
import torch
from torch.cuda import device_count
import numpy as np





import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
from shared.embedding_utils import split_sentences, embed_text, run_parallel_embeddings
def get_embedding(model_name, dataset, model_name_or_path, embedding_output_dir):
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

 # 1. Determine full output file path
 # Create dir first to ensure path exists for checking (,os.path.exists False)
 #, makedirs, 
 os.makedirs(embedding_output_dir, exist_ok=True) # Ensure directory exists for file checking
 output_file = os.path.join(embedding_output_dir, f"{model_name}.npy")

 # 2. Check if file exists
 if os.path.exists(output_file):
 print(f"Found existing embedding file: {output_file}")
 print("Loading embeddings...")
 try:
 results = np.load(output_file, allow_pickle=True) # allow_pickle=True 
 print("Embeddings loaded successfully.")
 return results
 except Exception as e:
 print(f"load file {output_file} error: {e}")
 print("Will recompute embeddings.")
 # If loading fails, continue with computation

 # 3. If file missing or load failed, compute embeddings
 print(f"Embedding file not found or unable to load: {output_file}")
 print("Starting embedding computation...")

 engine_count = device_count()

 print('Embedding model name:', model_name)

 # Load dataset texts
 texts_to_embed = dataset['text']
 if not texts_to_embed:
 print("Warning: Input dataset's 'text' field is empty.")
 return np.array([])

 # Set up multiple engines (one for each GPU)
 array = AsyncEngineArray.from_args([
 EngineArgs(
 batch_size= 32 if 'qwen' in model_name else 256,
 model_name_or_path=model_name_or_path,
 engine="torch",
 embedding_dtype="float32", # float32
 dtype="auto", # 
 device='cuda',
 device_id=f'{i}',
 served_model_name=f'model_{i}'
) for i in range(engine_count)
])

 
 # Run parallel embedding computation
 print("Running embedding computation...")
 results = asyncio.run(run_parallel_embeddings(array, texts_to_embed, engine_count))
 print("Embedding computation complete.")

 print(f"Embedding results preview (shape): {results.shape if isinstance(results, np.ndarray) else 'not a Numpy array'}") # 
 torch.cuda.empty_cache() 
 # 4. Save newly computed embeddings
 print(f"Saving embeddings to: {output_file}")
 try:
 # Ensure directory exists (if not created earlier)
 os.makedirs(embedding_output_dir, exist_ok=True)
 np.save(output_file, results)
 print("Embeddings saved successfully.")
 except Exception as e:
 print(f"save file {output_file} error: {e}")
 # Return results even if save fails
 
 return results



