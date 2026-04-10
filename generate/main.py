from functools import partial

from transformers import AutoTokenizer
import numpy as np
import json
import datasets
import re
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from torch.cuda import device_count
from arg_parser import parse_args
from tqdm import tqdm
from map_func import *
from loguru import logger
import torch
from shared.llm_utils import shutdown_vllm as shutdown, initialize_llm, run_generate, extract_text_and_tokens


os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0' # For A100 only, see https://en.wikipedia.org/wiki/CUDA
os.environ['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN']='1'


def generate(llm, dataset, sampling_params, tokenizer):
    """Thin wrapper around shared.llm_utils.run_generate."""
    return run_generate(llm, dataset, sampling_params, tokenizer, args.inference_backend)


def initialize(model_name, model_path):
    """Thin wrapper around shared.llm_utils.initialize_llm."""
    return initialize_llm(model_name, model_path, args.inference_backend)


def extract_from_outputs(outputs, tokenizer):
    """Thin wrapper around shared.llm_utils.extract_text_and_tokens."""
    return extract_text_and_tokens(outputs, tokenizer, args.inference_backend)

if __name__ == "__main__":
    args = parse_args()
    query_model_name = args.query_model_name
    output_dir = args.output_dir
    max_turn = int(args.turn)   # <-- max number of turns
    start = int(args.start)
    end = int(args.end)

    print(args)

    # ===== Detect which turn has been generated so far =====
    exp_dir = os.path.join(output_dir, f'{args.query_model_name}-{args.response_model_name}')
    if os.path.exists(exp_dir):
        existing_turns = []
        for d in os.listdir(exp_dir):
            if not d.isdigit():
                continue
            turn_dir = os.path.join(exp_dir, d, f"{start}-{end}")
            if os.path.exists(os.path.join(turn_dir, "dataset_info.json")):
                existing_turns.append(int(d))
        current_turn = max(existing_turns) if existing_turns else 0
    else:
        os.makedirs(exp_dir, exist_ok=True)
        current_turn = 0

    if current_turn >= max_turn:
        logger.info(f"Already generated turn {current_turn}, target {max_turn} turns, no need to continue.")
        exit(0)

    logger.info(f"Detected progress at turn {current_turn}, will continue generating turns {current_turn + 1} ~ {max_turn}")

    # ===== Initialize backend =====
    if args.inference_backend == 'sglang':
        import sglang as sgl
        sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 8192}
    elif args.inference_backend == 'vllm':
        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=8192,
            stop_token_ids=[255022],
        )
    else:
        logger.error(f"Please set correct inference_backend, current value: {args.inference_backend}")
        assert False

    # ===== Initialize tokenizer and model =====
    query_tokenizer = AutoTokenizer.from_pretrained(args.query_model_path)
    llm = initialize(args.query_model_name, args.query_model_path)

    # ===== Safe add_column wrapper =====
    def safe_add_column(dataset, name, data):
        if name in dataset.column_names:
            dataset = dataset.remove_columns([name])
        return dataset.add_column(name=name, column=data)

    # ===== Main loop, continue from next turn =====
    for turn in range(current_turn + 1, max_turn + 1):
        logger.info(f"\n========== Starting generation of turn {turn} ==========")
        output_path = os.path.join(output_dir, f'{args.query_model_name}-{args.response_model_name}', str(turn), f'{start}-{end}')
        cache_dir = os.path.join(args.cache_dir, args.query_model_name, str(turn), f'{start}-{end}')
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        # If this turn already exists, skip it
        if os.path.exists(os.path.join(output_path, "dataset_info.json")):
            logger.info(f"Turn {turn} already exists, skipping")
            continue

        # ====== Load or construct data ======
        try:
            dataset = datasets.load_from_disk(cache_dir)
            logger.info(f"Reusing cache {cache_dir}")
        except Exception:
            if turn == 1:
                dataset = datasets.load_from_disk(args.dataset_path).select(range(start, end))
            else:
                last_turn_dataset = os.path.join(
                    output_dir,
                    f'{args.query_model_name}-{args.response_model_name}',
                    f'{turn - 1}',
                    f'{start}-{end}',
                )
                dataset = datasets.load_from_disk(last_turn_dataset)

            # Generate query
            dataset = dataset.map(partial(make_query, tokenizer=query_tokenizer), num_proc=1, load_from_cache_file=False)
            outputs = generate(llm, dataset, sampling_params, query_tokenizer)
            output_text, completion_tokens = extract_from_outputs(outputs, query_tokenizer)

            dataset = safe_add_column(dataset, 'completion_tokens', completion_tokens)
            dataset = safe_add_column(dataset, 'raw_question', output_text)
            dataset = dataset.map(split_query, load_from_cache_file=False, remove_columns=['selected_did','input_ids','raw_question'])
            dataset = dataset.filter(lambda x: x['ground_truth_document_idx'] != -1, num_proc=64)
            dataset.save_to_disk(cache_dir)

        # ====== Response phase ======
        if args.query_model_name != args.response_model_name:
            shutdown(llm)
            llm = initialize(args.response_model_name, args.response_model_path)

        try:
            dataset = datasets.load_from_disk(output_path)
            logger.info(f"Turn {turn} already fully generated, skipping")
            continue
        except Exception:
            pass

        response_tokenizer = AutoTokenizer.from_pretrained(args.response_model_path)
        dataset = dataset.map(partial(make_response, tokenizer=response_tokenizer), num_proc=1)
        outputs = generate(llm, dataset, sampling_params, response_tokenizer)
        output_text, completion_tokens = extract_from_outputs(outputs, response_tokenizer)

        dataset = dataset.map(partial(merge_by_add, key='completion_tokens', data=completion_tokens), with_indices=True)

        formatted = [
            {
                'role': 'assistant',
                'content': t.split('\n</think>\n\n')[-1] if '\n</think>\n\n' in t else t
            }
            for t in output_text
        ]
        dataset = dataset.map(partial(merge_by_append, key='messages', data=formatted),
                              with_indices=True, remove_columns=['input_ids', 'query'])

        dataset.save_to_disk(output_path)
        logger.info(f"Turn {turn} generation complete, saved to {output_path}")
        torch.cuda.empty_cache()

    logger.info("All turns generation complete!")

