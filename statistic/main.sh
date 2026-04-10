model='unsloth/Llama-3.3-70B-Instruct'
processed_dataset_path='${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/processed_dataset'
sharegpt_path='${DATASETS_DIR:-/path/to/datasets}/liyucheng/ShareGPT90K'
domain_result_path='${MTR_ROOT:-/path/to/mtr}/tempfile'
coral_document_dir='${HOME_DIR:-/path/to}/datasets/coral/passage_corpus.json'
coral_dataset_dir='${HOME_DIR:-/path/to}/datasets/coral/test/test_conversation.json'
domain_classifier_path=nvidia/domain-classifier
mtr_test_path='${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/mtr_test'
mtr_train_path='${MTR_ROOT:-/path/to/mtr}/mtr-data-dumps/mtr_train'
python ${MTR_ROOT:-/path/to/mtr}/statistic.py --model $model --processed_dataset_path $processed_dataset_path --sharegpt_path $sharegpt_path --domain_classifier_path $domain_classifier_path --domain_result_path $domain_result_path --coral_document_dir $coral_document_dir --coral_dataset_dir $coral_dataset_dir --mtr_test_path $mtr_test_path --mtr_train_path $mtr_train_path