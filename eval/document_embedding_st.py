import logging
import os
import numpy as np
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.models import Pooling,Transformer
import argparse
import torch # Added for torch.cuda.empty_cache()
from torch.cuda import device_count
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
from shared.data_utils import parse_dataset_with_truncation as parse_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Embedding Generation Script')

    # Arguments adapted from example
    parser.add_argument('--embedding_output_path', type=str, required=True)
    parser.add_argument('--embedding_model_path', type=str, required=True, help="Comma-separated list of model paths")
    parser.add_argument('--embedding_model_name', type=str, required=True, help="Comma-separated list of model names")
    parser.add_argument('--dataset_path', type=str, required=True, help="Comma-separated list of dataset paths")
    parser.add_argument('--dataset_name', type=str, required=True, help="Comma-separated list of dataset names")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    embedding_model_path_list = args.embedding_model_path.split(',')
    embedding_model_name_list = args.embedding_model_name.split(',')
    dataset_path_list = args.dataset_path.split(',')
    dataset_name_list = args.dataset_name.split(',')

    if not (len(embedding_model_path_list) == len(embedding_model_name_list)):
        raise ValueError("The number of embedding model paths and names must be the same.")
    if not (len(dataset_path_list) == len(dataset_name_list)):
        raise ValueError("The number of dataset paths and names must be the same.")

    for model_idx, (current_model_path, current_model_name) in enumerate(zip(embedding_model_path_list, embedding_model_name_list)):
        logging.info(f"Processing model: {current_model_name} from path: {current_model_path}")

        # Initialize model
        if 'qwen' in current_model_name.lower():
            model = SentenceTransformer(current_model_path, trust_remote_code=True, model_kwargs={'torch_dtype':'auto'})
        else:
            transformer_model = Transformer(current_model_path, model_args={'trust_remote_code':True})
            pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode='cls')
            model = SentenceTransformer(modules=[transformer_model, pooling_model])
        
        # Start multi-process pool for the current model
        pool = model.start_multi_process_pool()
        logging.info(f"Started multi-process pool for model {current_model_name}")

        for dataset_idx, (current_dataset_path, current_dataset_name) in enumerate(zip(dataset_path_list, dataset_name_list)):
            logging.info(f"Processing dataset: {current_dataset_name} from path: {current_dataset_path} for model: {current_model_name}")

            embedding_output_dir_for_dataset = os.path.join(args.embedding_output_path, current_dataset_name)
            os.makedirs(embedding_output_dir_for_dataset, exist_ok=True)
            
            # Output file name convention from example
            output_file = os.path.join(embedding_output_dir_for_dataset, f"document_{current_model_name}.npy")

            if os.path.exists(output_file):
                logging.info(f"Skipping. Found existing embeddings: {output_file}")
                continue

            # Load dataset
            logging.info(f"Loading dataset: {current_dataset_name}")
            dataset = parse_dataset(current_dataset_path, current_dataset_name)
            
            if not dataset or 'text' not in dataset.column_names or len(dataset) == 0:
                logging.error(f"Dataset {current_dataset_name} is empty or does not contain 'text' column. Skipping.")
                continue


            # texts_to_embed = dataset['text'] # This is handled by DataLoader later
            # if not texts_to_embed:
            #     logging.error(f"Input dataset '{current_dataset_name}' 'text' field is empty. Skipping.")
            #     continue


            # Set DataLoader
            # Consider adjusting batch_size and num_workers based on available resources
            # The batch_size for dataloader can be large as it's for IO.
            # The batch_size for model.encode_multi_process is for GPU processing.
            dataloader_batch_size = 8192*device_count() # Example: process 8192 sentences from disk at a time
            encode_batch_size = 1024 if 'qwen' not in current_model_name.lower() else 16
            encode_chunk_size = 1024*device_count() if 'qwen' not in current_model_name.lower() else 16*device_count() # chunk_size for sentence_transformers

            dataloader = DataLoader(
                dataset=dataset, # dataset is an IterableDataset or MapStyleDataset from `datasets`
                batch_size=dataloader_batch_size,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=4,
            )

            all_embeddings_list = [] # Use a list to append numpy arrays

            logging.info(f"Starting embedding generation for {current_dataset_name} with model {current_model_name}")
            for i, batch in enumerate(tqdm(dataloader, desc=f"Embedding {current_dataset_name}")):
                # The 'batch' from datasets DataLoader will be a dictionary of lists/tensors
                if "title" in batch and "text" in batch:
                    sentences = [f"{t}\n{x}" for t, x in zip(batch["title"], batch["text"])]
                elif "text" in batch:
                    sentences = batch["text"]
                else:
                    logging.error(f"Batch {i+1} does not contain 'text' field. Skipping batch.")
                    continue
                
                if not sentences:
                    logging.warning(f"Batch {i+1} resulted in empty sentences list. Skipping batch.")
                    continue

                try:
                    batch_emb = model.encode_multi_process(
                        sentences, 
                        pool, 
                        batch_size=encode_batch_size,
                        chunk_size=encode_chunk_size 
                    )

                    if np.any(np.all(batch_emb == 0, axis=1)):
                        logging.warning(f"All-zero embeddings found in batch {i+1} for dataset {current_dataset_name}.")
                        # Potentially add more debugging here if needed
                        # e.g., print some sentences that resulted in zero embeddings

                    # logging.debug(f"Embeddings computed for batch {i+1}. Shape: {batch_emb.shape}")
                    all_embeddings_list.append(batch_emb)
                except Exception as e:
                    logging.error(f"Error encoding batch {i+1} for {current_dataset_name}: {e}")
                    # Decide if you want to skip or halt
                    # For now, just log and continue to next batch
                    continue
            
            if not all_embeddings_list:
                logging.error(f"No embeddings were generated for dataset {current_dataset_name}. Skipping save.")
                continue

            all_embeddings = np.vstack(all_embeddings_list)
            logging.info(f"Total embeddings shape for {current_dataset_name}: {all_embeddings.shape}")

            logging.info(f"Saving embeddings to: {output_file}")
            np.save(output_file, all_embeddings)
            logging.info(f"Embeddings for {current_dataset_name} saved at {output_file}")

            if np.any(np.all(all_embeddings == 0, axis=1)):
                logging.warning(f"All-zero embeddings found in final output for {current_dataset_name}.")

        # Stop multi-process pool after processing all datasets for the current model
        logging.info(f"Stopping multi-process pool for model {current_model_name}")
        model.stop_multi_process_pool(pool)
        
        # Optional: Clear GPU cache if using GPU, after each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"Cleared CUDA cache after processing model {current_model_name}")


    logging.info("All models and datasets have been processed.")