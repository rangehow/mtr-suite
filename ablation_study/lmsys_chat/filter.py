import argparse
import torch
import os
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# System Prompt
SYSTEM_PROMPT = """You are an expert data analyst. Your task is to classify a conversation as "knowledge-intensive" or "not knowledge-intensive".

A "knowledge-intensive" conversation is one where the user is genuinely seeking factual information or explanation about the world, such as history, science, technology, art, culture, etc. The focus is on acquiring established knowledge.

The following are NOT "knowledge-intensive":
- Coding or programming questions.
- Mathematical problem-solving.
- Creative writing, poetry, or storytelling.
- Simple greetings, chit-chat, or personal opinions without a factual basis.
- Instructions to summarize, translate, or rephrase text without seeking new knowledge."""

# User Prompt, 
USER_PROMPT_WRAPPER = """Please classify the following conversation. Respond with only "Yes" or "No".

--- Conversation to Classify ---
{conversation_text}
--- End of Conversation ---"""

def format_conversation_to_string(conversation_list):
 """,."""
 return "\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation_list)

def main(args):
 # 
 if not torch.cuda.is_available():
 raise EnvironmentError(" CUDA, GPU.")
 tensor_parallel_size = torch.cuda.device_count()
 print(f" {tensor_parallel_size} GPU,.")

 # --- 1. Tokenizer ---
 print(f" {args.dataset_path}...")
 if not os.path.exists(args.dataset_path):
 raise FileNotFoundError(f": {args.dataset_path}")
 dataset = load_dataset(args.dataset_path)['train']
 
 print(f" Tokenizer: {args.model_id}...")
 tokenizer = AutoTokenizer.from_pretrained(args.model_id)
 print(f" Tokenizer.: {len(dataset)}")

 print(f" ( > {args.multi_turn_threshold})...")
 multi_turn_data = dataset.filter(
 lambda x: len(x['conversation']) > args.multi_turn_threshold and len(x['conversation']) % 2 == 0,
 num_proc=os.cpu_count()
)
 print(f" {len(multi_turn_data)}.")

 if len(multi_turn_data) == 0:
 print(",.")
 return

 # --- 3. [] map Prompts ---
 #, map 
 def format_and_tokenize_example(example):
 conversation_str = format_conversation_to_string(example['conversation'])
 classification_messages = [
 {"role": "system", "content": SYSTEM_PROMPT},
 {"role": "user", "content": USER_PROMPT_WRAPPER.format(conversation_text=conversation_str)}
]
 prompt_tokens = tokenizer.apply_chat_template(
 classification_messages,
 tokenize=True,
 add_generation_prompt=True
)
 prompt = tokenizer.apply_chat_template(
 classification_messages,
 tokenize=False,
 add_generation_prompt=True
)
 return {"prompt_tokens": prompt_tokens, "prompt_length": len(prompt_tokens),'prompt':prompt}

 print(" map tokenize...")
 # map prompt_tokens prompt_length 
 processed_data = multi_turn_data.map(
 format_and_tokenize_example,
 num_proc=64,
 desc=" Tokenize"
)

 # --- 4. [] filter ---
 print(f" (<= {args.max_prompt_length} tokens)...")
 original_count = len(processed_data)
 filtered_by_length_data = processed_data.filter(
 lambda x: x['prompt_length'] <= args.max_prompt_length,
 num_proc=os.cpu_count()
)
 print(f". {original_count}, {len(filtered_by_length_data)} prompts.")

 if len(filtered_by_length_data) == 0:
 print(",.")
 return

 # --- 5. vLLM ---
 print(f" vLLM: {args.model_id}...")
 llm = LLM(
 model=args.model_id,
 tensor_parallel_size=tensor_parallel_size,
 trust_remote_code=True,
 max_model_len=args.max_prompt_length, # 
)
 sampling_params = SamplingParams(temperature=0, max_tokens=10) # 

 # tokenized prompts 
 prompts_to_process = filtered_by_length_data['prompt']

 print(f" {len(prompts_to_process)} vLLM,...")
 outputs = llm.generate(prompts_to_process, sampling_params=sampling_params,use_tqdm=True)
 print("vLLM.")

 # --- 6. ---
 classifications = [output.outputs[0].text.strip().lower() for output in outputs]
 final_data_before_filter = filtered_by_length_data.add_column("classification_result", classifications)

 # 
 print(" LLM ('yes')...")
 final_dataset = final_data_before_filter.filter(
 lambda x: x['classification_result'].startswith("yes"),
 num_proc=os.cpu_count()
)
 print(f". {len(final_dataset)}.")
 
 if len(final_dataset) == 0:
 print(".")
 return
 
 final_dataset = final_dataset.remove_columns(['prompt_tokens', 'prompt_length', 'classification_result'])

 # --- 7. ---
 print(f" ({len(final_dataset)}) {args.output_path}...")
 final_dataset.save_to_disk(args.output_path)
 
 print("!")

if __name__ == "__main__":
 parser = argparse.ArgumentParser(description=" LLM.")
 parser.add_argument("--dataset_path", type=str, required=True, help=" Hugging Face.")
 parser.add_argument("--output_path", type=str, required=True, help=".")
 parser.add_argument("--model_id", type=str, required=True, help=" VLLM ID.")
 parser.add_argument("--multi_turn_threshold", type=int, default=3, help=".")
 parser.add_argument("--max_prompt_length", type=int, default=40960, help=" prompt token.")
 
 cli_args = parser.parse_args()
 main(cli_args)