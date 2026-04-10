source ${HOME_DIR:-/path/to}/.bashrc
mamba activate sglang

wiki_2025_dump="${DATASETS_DIR:-/path/to/datasets}/NeuML/wikipedia-20250123"
save_dir="${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/processed_dataset"
QUALITY_MODEL_PATH="nvidia/quality-classifier-deberta"
FINEWEB_MODEL_PATH="nvidia/nemocurator-fineweb-nemotron-4-edu-classifier"
python ${MTR_ROOT:-/path/to/mtr}/data_process.py --dataset_path $wiki_2025_dump --save_dir $save_dir --quality_model_path $QUALITY_MODEL_PATH --fineweb_model_path $FINEWEB_MODEL_PATH