source ${HOME_DIR:-/path/to}/.bashrc
mamba activate sglang

chatrag_dataset='${HOME_DIR:-/path/to}/datasets/synthethisqa'
mtr_dataset='${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/mtr_train'
output_dir="${MTR_ROOT:-/path/to/mtr}/tempfile/model"


model_dir='Alibaba-NLP/gte-modernbert-base'


# save_name='chatrag_moderbert_base'
# train_dataset_name='chatrag'


save_name='mtr_moderbert_base'
train_dataset_name='mtr'

torchrun --nproc-per-node 4 ${MTR_ROOT:-/path/to/mtr}/train/train.py --train_dataset_name $train_dataset_name --save_name $save_name --model_dir $model_dir --chatrag_dataset $chatrag_dataset --mtr_dataset $mtr_dataset --output_dir $output_dir