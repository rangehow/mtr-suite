import argparse

def parse_args():
 """parseparameter"""
    parser = argparse.ArgumentParser(description="Process datasets with padding and multiprocessing.")
    # embedding begin -------
    parser.add_argument("--embedding_model", help="Model name or path.")
    parser.add_argument("--processed_dataset_path", help="Path to the dataset.")
    parser.add_argument("--embedding_output_dir", help="Directory to save embeddings.")
    # embedding end   ------------------------
    parser.add_argument("--index_output_dir", help="Directory to save faiss index.")
    parser.add_argument("--model_name")
    parser.add_argument('--topk', type=int, required=True, help='The topK value')
    parser.add_argument('--query_batch_size', type=int, default=2097152, help='too large can lead to oom')
    # cluster start

    parser.add_argument("--cluster_dataset_output_dir",type=str,required=True)
    parser.add_argument("--cluster_size",type=int,required=True)
    return parser.parse_args()