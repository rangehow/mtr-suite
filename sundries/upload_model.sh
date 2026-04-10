export HF_TOKEN='your_hf_token_here'
model_path='${MTR_ROOT:-/path/to/mtr}/tempfile/model/chatrag_moderbert_base/checkpoint-52'
hf_repo_name='your-org/your-model-name'
python ${MTR_ROOT:-/path/to/mtr}/sundries/upload_model.py --model_path $model_path --hf_repo_name $hf_repo_name