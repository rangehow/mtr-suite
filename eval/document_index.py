
import os
import numpy as np
from loguru import logger

import faiss
import argparse




def parse_args():
    parser = argparse.ArgumentParser(description='Embedding Generation Script')
    
    parser.add_argument('--embedding_output_path', type=str, required=True)
    parser.add_argument('--index_output_path', type=str, required=True)
 # --- receiveparameterstring ---
    parser.add_argument('--embedding_model_name', type=str, required=True)

    parser.add_argument('--dataset_name', type=str, required=True)

    

    return parser.parse_args()


if __name__=='__main__':

    args = parse_args()
    

    embedding_model_name_list = args.embedding_model_name.split(',')

    dataset_name_list = args.dataset_name.split(',')

    
    for model_name in embedding_model_name_list:

        

        for dataset_name in dataset_name_list:
            embedding_output_dir = os.path.join(args.embedding_output_path,dataset_name)
            embedding_file = os.path.join(embedding_output_dir, f"document_{model_name}.npy")

            index_save_path = os.path.join(args.index_output_path,dataset_name)
            os.makedirs(index_save_path,exist_ok=True)
            index_save_path = os.path.join(index_save_path,f'{model_name}.idx')

            if os.path.exists(index_save_path):
 logger.success(f"existsindexfile: {index_save_path}")
            
            else:
                logger.info(dataset_name)
                try:
                    embedding = np.load(embedding_file)
                except Exception as e:
                    logger.error(f'{e}')
                

                dim = embedding.shape[-1]
                logger.info(f'current dataset{dataset_name},embedding shape:{embedding.shape}')
                index = faiss.IndexFlatIP(dim)
                index.add(embedding)

                # 4. Save newly computed embeddings
 logger.info(f"currentlysaveindex: {index_save_path}")
                faiss.write_index(index, index_save_path)
