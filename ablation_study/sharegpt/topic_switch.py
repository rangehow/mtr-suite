import argparse
import json
import os
import time
from typing import List, Dict

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = """You are an expert conversation analyst. Your task is to determine if a user's latest message introduces a significant topic switch away from the preceding conversation.
Analyze the conversation history and the user's latest turn. You MUST respond in a strict JSON format.

Your JSON output must contain two keys:
{
 "reason": "string", // A brief explanation for your decision.
 "is_topic_switch": boolean // Set to true if it's a topic switch, otherwise false.
}

Guidelines for your decision:
- A generalization from a specific topic to a broader one (e.g., from 'how to grill a steak' to 'what are your cooking skills') can be considered a topic switch.
- A completely unrelated topic (e.g., switching from 'Python programming' to 'what's the weather like today') is a clear topic switch."""

# token ( `[INST]`, `\n`, `<s>`)
TEMPLATE_TOKEN_BUFFER_PER_MESSAGE = 10

#, dataset.map 
tokenizer = None
role_mapping = {'human': 'user', 'gpt': 'assistant'}
MODEL_MAX_LENGTH = 40960
SAFE_MAX_LENGTH = int(MODEL_MAX_LENGTH * 0.9)


def generate_prompts_for_conversation_batched(examples: Dict[str, List]):
 """
, prompts.

 “ ”,,.
,, tokenization.
, prompt.

 Args:
 examples (Dict[str, List]):
, Hugging Face Datasets, 'id' 'conversations'.
 'conversations':
 1. List[Dict]: [{'from': 'human', 'value': '...'}, {'from': 'gpt', 'value': '...'}]
 2. Dict[str, List]: {'from': ['human', 'gpt'], 'value': ['...', '...']}

 Returns:
 Dict[str, List[List[Any]]]:
, 'prompts' 'metadata'.
 'prompts':,, prompts.
 'metadata': 'prompts', prompt.
 """
 # 
 global tokenizer, role_mapping, SYSTEM_PROMPT, SAFE_MAX_LENGTH, TEMPLATE_TOKEN_BUFFER_PER_MESSAGE

 batch_prompts = []
 batch_metadata = []

 conv_ids = examples['id']
 all_conversations_in_batch = examples['conversations']

 for i in range(len(conv_ids)):
 conv_id = conv_ids[i]
 messages_raw = all_conversations_in_batch[i]

 prompts_for_one_conv = []
 metadata_for_one_conv = []

 if isinstance(messages_raw, dict) and 'from' in messages_raw and 'value' in messages_raw:
 roles = messages_raw.get('from', [])
 values = messages_raw.get('value', [])
 messages = [{"from": r, "value": v} for r, v in zip(roles, values)]
 elif isinstance(messages_raw, list):
 messages = messages_raw
 else:
 batch_prompts.append([])
 batch_metadata.append([])
 continue

 all_texts = [SYSTEM_PROMPT] + [msg.get('value', '') for msg in messages]
 try:
 # add_special_tokens=False token 
 token_lengths = [len(enc) for enc in tokenizer(all_texts, add_special_tokens=False).input_ids]
 except Exception as e:
 print(f"Warning: Tokenization failed for conv_id {conv_id}. Skipping. Error: {e}")
 batch_prompts.append([])
 batch_metadata.append([])
 continue
 
 system_prompt_len = token_lengths[0]
 message_lengths = token_lengths[1:]

 #, "user" prompt
 for msg_idx in range(len(messages)):
 current_turn = messages[msg_idx]
 current_role_raw = current_turn.get('from')
 current_role = role_mapping.get(current_role_raw)

 # user prompt
 if current_role!= 'user':
 continue

 current_content = current_turn.get('value', '')
 if not current_content:
 continue
 

 # 1. “ ”:System Prompt + User + token 
 current_turn_len = message_lengths[msg_idx]
 current_total_len = system_prompt_len + current_turn_len + (TEMPLATE_TOKEN_BUFFER_PER_MESSAGE * 2)

 # “ ”.,, prompt.
 if current_total_len > SAFE_MAX_LENGTH:
 # print(f"Warning: Skipping message {msg_idx} in conv_id {conv_id} because its base length ({current_total_len}) exceeds the safe limit ({SAFE_MAX_LENGTH}).")
 continue

 # 2.,, 
 history_indices_to_include = []
 for hist_idx in range(msg_idx - 1, -1, -1):
 history_msg_len = message_lengths[hist_idx]
 
 # 
 if current_total_len + history_msg_len + TEMPLATE_TOKEN_BUFFER_PER_MESSAGE > SAFE_MAX_LENGTH:
 break
 
 current_total_len += history_msg_len + TEMPLATE_TOKEN_BUFFER_PER_MESSAGE
 history_indices_to_include.append(hist_idx)


 history_indices_to_include.reverse()

 # 4. payload prompt
 final_history = [
 {"role": role_mapping.get(messages[j]['from']), "content": messages[j]['value']}
 for j in history_indices_to_include
]
 
 current_turn_payload = [
 {"role": "system", "content": SYSTEM_PROMPT}
] + final_history + [
 {"role": "user", "content": current_content}
]

 final_prompt_str = tokenizer.apply_chat_template(
 current_turn_payload, tokenize=False, add_generation_prompt=True
)
 
 prompts_for_one_conv.append(final_prompt_str)
 metadata_for_one_conv.append({
 "conv_id": conv_id, 
 "message_index": msg_idx, 
 "message_content": current_content
 })
 batch_prompts.append(prompts_for_one_conv)
 batch_metadata.append(metadata_for_one_conv)

 return {"prompts": batch_prompts, "metadata": batch_metadata}


def parse_arguments():
 """ """
 parser = argparse.ArgumentParser(description="Detect topic switches in conversations using vLLM.")
 parser.add_argument("--dataset_path", type=str, required=True, help="Path to the directory containing the ShareGPT JSON files.")
 parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSONL file.")
 parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use from Hugging Face Hub.")
 parser.add_argument("--max_conversations", type=int, default=None, help="Maximum number of conversations to process. Default is all.")
 parser.add_argument("--num_proc", type=int, default=None, help="Number of processes for dataset.map(). Defaults to available CPUs.")
 parser.add_argument("--map_batch_size", type=int, default=1000, help="Batch size for dataset.map() processing.")
 return parser.parse_args()

def load_sharegpt_dataset(path: str, max_samples: int = None):
 """ ShareGPT """
 dataset = load_dataset(path, split='train')
 if max_samples:
 return dataset.select(range(min(max_samples, len(dataset))))
 return dataset

def process_and_save_results(outputs, metadata_list: List[Dict], output_file: str):
 """ vLLM """
 print(f" {len(outputs)} '{output_file}'...")
 switch_count = 0
 with open(output_file, 'w', encoding='utf-8') as f:
 for i, output in enumerate(tqdm(outputs, desc="Parsing Results")):
 metadata = metadata_list[i]
 generated_text = output.outputs[0].text.strip()
 if generated_text and not generated_text.startswith('{'): generated_text = '{' + generated_text
 if generated_text and not generated_text.endswith('}'): generated_text += '}'
 try:
 analysis_result = json.loads(generated_text)
 if analysis_result.get("is_topic_switch", False):
 switch_count += 1
 result_entry = {
 "conv_id": metadata["conv_id"],
 "message_index": metadata["message_index"],
 "content": metadata["message_content"],
 "is_topic_switch": True,
 "reason": analysis_result.get("reason", "N/A")
 }
 f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
 except json.JSONDecodeError:
 print(f"\n: JSON.: {metadata},: '{generated_text}'")
 print(f". {switch_count}.")
 return switch_count

def main():
 """ """
 global tokenizer, MODEL_MAX_LENGTH, SAFE_MAX_LENGTH
 
 args = parse_arguments()
 start_time = time.time()
 
 if not torch.cuda.is_available():
 print(": CUDA is not available. vLLM GPU.")
 exit(1)
 tensor_parallel_size = torch.cuda.device_count()
 print(f" {tensor_parallel_size} GPU, tensor_parallel_size={tensor_parallel_size}")

 print("...")
 tokenizer = AutoTokenizer.from_pretrained(args.model_name)

 SAFE_MAX_LENGTH = int(MODEL_MAX_LENGTH * 0.9)
 print(f" Token: {MODEL_MAX_LENGTH},: {SAFE_MAX_LENGTH}")

 dataset = load_sharegpt_dataset(args.dataset_path, args.max_conversations)
 num_conversations = len(dataset)
 
 num_proc = args.num_proc if args.num_proc else os.cpu_count()
 print(f" dataset.map prompts ( {num_proc},: {args.map_batch_size})...")
 
 # ---: batched=True ---
 processed_dataset = dataset.map(
 generate_prompts_for_conversation_batched,
 batched=True,
 batch_size=args.map_batch_size,
 num_proc=num_proc,
 
 desc="Generating prompts in batched mode"
)
 
 print(" prompts metadata...")
 all_prompts = [prompt for example in tqdm(processed_dataset, desc="Flattening prompts") for prompt in example['prompts']]
 all_metadata = [meta for example in tqdm(processed_dataset, desc="Flattening metadata") for meta in example['metadata']]
 
 if not all_prompts:
 print(",..")
 return
 
 print(f" {num_conversations} {len(all_prompts)}.")

 print(" vLLM...")
 llm = LLM(
 model=args.model_name, 
 trust_remote_code=True, 
 tensor_parallel_size=tensor_parallel_size,
 max_model_len=MODEL_MAX_LENGTH,
 enable_prefix_caching=True
)
 print(".")

 sampling_params = SamplingParams(temperature=0.0, max_tokens=150, stop=["}"])
 print(" vLLM...")
 inference_start_time = time.time()
 outputs = llm.generate(all_prompts, sampling_params)
 print(f",: {time.time() - inference_start_time:.2f}.")

 switch_count = process_and_save_results(outputs, all_metadata, args.output_file)
 
 total_time = time.time() - start_time
 print("\n--- ---")
 print(f": {total_time:.2f} ")
 print(f": {num_conversations}")
 print(f": {len(all_prompts)}")
 print(f": {switch_count}")
 if len(all_prompts) > 0:
 switch_rate = (switch_count / len(all_prompts)) * 100
 print(f" (): {switch_rate:.2f}%")
 
if __name__ == "__main__":
 main()