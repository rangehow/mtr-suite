# source ${HOME_DIR:-/path/to}/.bashrc
# mamba activate sglang

# embedding_model_dir='intfloat/multilingual-e5-large-instruct'
embedding_model_dir=Qwen/Qwen3-Embedding-8B


python ${MTR_ROOT:-/path/to/mtr}/embedding.py --embedding_model $embedding_model_dir --model_name gte-qwen3-8b --embedding_output_dir ${MTR_ROOT:-/path/to/mtr}/tempfile/embedding_health --processed_dataset_path ${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/health_document_subset --topk 100 --index_output_dir ${MTR_ROOT:-/path/to/mtr}/tempfile/index_health --cluster_dataset_output_dir ${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/cluster_dataset_health --cluster_size 8


# python ${MTR_ROOT:-/path/to/mtr}/embedding.py --embedding_model $embedding_model_dir --model_name me5-large --embedding_output_dir ${MTR_ROOT:-/path/to/mtr}/tempfile/embedding --processed_dataset_path ${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/processed_dataset --topk 50 --index_output_dir ${MTR_ROOT:-/path/to/mtr}/tempfile/index --cluster_dataset_output_dir ${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/cluster_dataset --cluster_size 8