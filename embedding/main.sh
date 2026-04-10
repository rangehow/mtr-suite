# source ${HOME_DIR:-/path/to}/.bashrc
# mamba activate sglang

# embedding_model_dir='intfloat/multilingual-e5-large-instruct'
# embedding_model_dir=Alibaba-NLP/gte-Qwen2-7B-instruct
embedding_model_dir=Qwen/Qwen3-Embedding-4B


embedding_output_dir=${MTR_ROOT:-/path/to/mtr}/tempfile/embedding/finance_unfiltered
processed_dataset_path=${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/processed_finance_unfiltered
index_output_dir=${MTR_ROOT:-/path/to/mtr}/tempfile/index/finance_unfiltered
cluster_dataset_output_dir=${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/cluster_dataset/finance_unfiltered


# embedding_output_dir=${MTR_ROOT:-/path/to/mtr}/tempfile/embedding/finance_filtered
# processed_dataset_path=${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/processed_finance_filtered
# index_output_dir=${MTR_ROOT:-/path/to/mtr}/tempfile/index/finance_filtered
# cluster_dataset_output_dir=${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/cluster_dataset/finance_filtered

python ${MTR_ROOT:-/path/to/mtr}/embedding.py --embedding_model $embedding_model_dir --model_name gte-qwen2 --embedding_output_dir ${embedding_output_dir} --processed_dataset_path ${processed_dataset_path}  --topk 100 --index_output_dir ${index_output_dir} --cluster_dataset_output_dir ${cluster_dataset_output_dir} --cluster_size 8


# python ${MTR_ROOT:-/path/to/mtr}/embedding.py --embedding_model $embedding_model_dir --model_name me5-large --embedding_output_dir ${MTR_ROOT:-/path/to/mtr}/tempfile/embedding --processed_dataset_path ${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/processed_dataset --topk 50 --index_output_dir ${MTR_ROOT:-/path/to/mtr}/tempfile/index --cluster_dataset_output_dir ${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/cluster_dataset --cluster_size 8