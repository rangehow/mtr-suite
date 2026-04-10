dataset_dir='${HOME_DIR:-/path/to}/datasets/coral/test/new_test_conversation.json'
document_dir='${HOME_DIR:-/path/to}/datasets/coral/passage_corpus.json'
output_dir='${HOME_DIR:-/path/to}/datasets/new_coral_hf'
python ${MTR_ROOT:-/path/to/mtr}/analysis_of_previous_benchmark/process_coral.py --dataset_dir $dataset_dir --document_dir $document_dir --output_dir $output_dir