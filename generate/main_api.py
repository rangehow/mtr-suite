"""
MTR dialogue generation using remote OpenAI-compatible API endpoints.

This is the API-backend counterpart of generate/main.py. Instead of loading
models locally with vLLM/SGLang, it sends requests to pre-deployed sglang
servers (e.g. 5× Qwen3.5-FP8 nodes).

Usage:
    python generate/main_api.py \
        --dataset_path mtr-data-dumps/cluster_dataset \
        --query_endpoints "http://<node1>:8080/v1/chat/completions,http://<node2>:8080/v1/chat/completions,..." \
        --model_id qwen35-fp8 \
        --model_display_name Qwen3.5-FP8 \
        --output_dir mtr-data-dumps/mtr \
        --start 0 --end 999 --turn 4 \
        --cache_dir tempfile/cache \
        --max_concurrent 80 \
        --enable_thinking false
"""

import argparse
import os
import random
import re
import sys
from functools import partial

import datasets
from loguru import logger
# Note: AutoTokenizer not needed for API-based generation (no local tokenization)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.api_utils import ApiBatchGenerator
from shared.map_utils import merge_by_add, merge_by_replace, merge_by_append

# Import prompts
from prompt import QUERY_WO_HISTORY, QUERY_W_HISTORY, RESPONSE
import seed_prompt


# ===== Argument parser =====

def parse_args():
    p = argparse.ArgumentParser(description="MTR generation via remote API endpoints")
    p.add_argument('--dataset_path', required=True, help='Path to clustered dataset')
    p.add_argument('--query_endpoints', required=True,
                   help='Comma-separated list of OpenAI-compatible chat/completions URLs')
    p.add_argument('--response_endpoints', default=None,
                   help='Separate endpoints for response generation (default: same as query)')
    p.add_argument('--model_id', default='qwen35-fp8',
                   help='Model ID as reported by the sglang server')
    p.add_argument('--model_display_name', default='Qwen3.5-FP8',
                   help='Human-readable model name (used in output path)')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--turn', type=int, required=True, help='Max number of turns to generate')
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--max_concurrent', type=int, default=80,
                   help='Max concurrent API requests across all endpoints')
    p.add_argument('--enable_thinking', default='false',
                   help='Enable model thinking/reasoning (true/false)')
    p.add_argument('--max_new_tokens', type=int, default=8192)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_p', type=float, default=0.95)
    return p.parse_args()


# ===== Query building (messages-based, no tokenizer needed for API) =====

def build_query_messages(instance):
    """Build the chat messages for query generation (1 instance)."""
    ctx = instance['ctxs']
    ctx_sample = random.sample(ctx, k=min(5, len(ctx)))

    document_list = [item['text'] for item in ctx_sample]
    cleaned_documents = [re.sub(r'\[\d+\]', '', doc) for doc in document_list]
    document_str = '\n---\n'.join([f"[{i+1}] {doc}" for i, doc in enumerate(cleaned_documents)])

    selected_did = [item['document_idx'] for item in ctx_sample]

    if 'messages' not in instance or not instance['messages']:
        SEED = random.choice(seed_prompt.QUERY_WO_HISTORY)
        prompt = QUERY_WO_HISTORY.format_map({'SEED': SEED, 'DOCUMENTS': document_str})
    else:
        SEED = random.choice(seed_prompt.CONVERSATION_W_HISTORY)
        history_str = ""
        previous_questions = []
        for message in instance['messages']:
            role_prefix = "User" if message['role'] == 'user' else "Assistant"
            history_str += f"{role_prefix}: {message['content']}\n"
            if message['role'] == 'user':
                previous_questions.append(message['content'])

        prev_q_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(previous_questions)]) if previous_questions else "(None)"

        prompt = QUERY_W_HISTORY.format_map({
            'SEED': SEED,
            'DOCUMENTS': document_str,
            'HISTORY': history_str.strip(),
            'PREVIOUS_QUESTIONS': prev_q_str
        })

    messages = [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': '**Question:**\n[', 'prefix': True},
    ]
    return messages, selected_did


def build_response_messages(instance):
    """Build the chat messages for response generation (1 instance)."""
    ground_truth_document = instance['ground_truth_ctx']['text']
    document_str = re.sub(r'\[\d+\]', '', ground_truth_document)
    question = instance['messages'][-1]['content']

    prompt = RESPONSE.format_map({'DOCUMENTS': document_str, 'QUESTION': question})

    # Prepend conversation history (all messages except the last user turn)
    history = instance['messages'][:-1]
    messages = history + [{'role': 'user', 'content': prompt}]
    return messages


def parse_query_output(text, selected_did, ctxs, existing_messages=None):
    """Parse the LLM query output into structured fields."""
    # Strip thinking tokens if present
    if '\n</think>\n\n' in text:
        text = text.split('\n</think>\n\n')[-1]

    pattern = r'(\d+)\]\s*(.+)'
    matches = re.findall(pattern, text)

    if not matches:
        logger.warning(f'No pattern match in: {text[:100]}')
        return None

    did_str, query = matches[0]
    query = query.strip()

    if not did_str.isdigit():
        return None

    idx = int(did_str) - 1
    if idx < 0 or idx >= len(selected_did):
        return None

    gt_doc_idx = selected_did[idx]
    gt_ctx = next((ctx for ctx in ctxs if ctx.get('document_idx') == gt_doc_idx), None)
    if gt_ctx is None:
        return None

    messages = (existing_messages or []) + [{'role': 'user', 'content': query}]

    return {
        'ground_truth_document_idx': gt_doc_idx,
        'ground_truth_ctx': gt_ctx,
        'query': query,
        'messages': messages,
    }


def strip_thinking(text):
    """Remove <think> blocks from response text."""
    if '\n</think>\n\n' in text:
        return text.split('\n</think>\n\n')[-1]
    return text


# ===== Safe add_column =====

def safe_add_column(dataset, name, data):
    if name in dataset.column_names:
        dataset = dataset.remove_columns([name])
    return dataset.add_column(name=name, column=data)


# ===== Main =====

if __name__ == "__main__":
    args = parse_args()

    enable_thinking = args.enable_thinking.lower() in ('true', '1', 'yes')

    query_endpoints = [e.strip() for e in args.query_endpoints.split(',') if e.strip()]
    response_endpoints = (
        [e.strip() for e in args.response_endpoints.split(',') if e.strip()]
        if args.response_endpoints else query_endpoints
    )

    logger.info(f"Query endpoints ({len(query_endpoints)}): {query_endpoints}")
    logger.info(f"Response endpoints ({len(response_endpoints)}): {response_endpoints}")
    logger.info(f"Model ID: {args.model_id}, Display name: {args.model_display_name}")
    logger.info(f"Enable thinking: {enable_thinking}")

    sampling_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_new_tokens': args.max_new_tokens,
    }

    query_gen = ApiBatchGenerator(
        endpoints=query_endpoints,
        model=args.model_id,
        max_concurrent=args.max_concurrent,
        enable_thinking=enable_thinking,
    )
    response_gen = ApiBatchGenerator(
        endpoints=response_endpoints,
        model=args.model_id,
        max_concurrent=args.max_concurrent,
        enable_thinking=enable_thinking,
    )

    model_tag = f"{args.model_display_name}-{args.model_display_name}"
    start, end = args.start, args.end
    max_turn = args.turn

    # ===== Detect existing progress =====
    exp_dir = os.path.join(args.output_dir, model_tag)
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
        logger.info(f"Already generated turn {current_turn}, target {max_turn}. Done.")
        exit(0)

    logger.info(f"Progress: turn {current_turn}, will generate turns {current_turn + 1} ~ {max_turn}")

    # ===== Main generation loop =====
    for turn in range(current_turn + 1, max_turn + 1):
        logger.info(f"\n========== Turn {turn} ==========")
        output_path = os.path.join(args.output_dir, model_tag, str(turn), f'{start}-{end}')
        cache_dir = os.path.join(args.cache_dir, args.model_display_name, str(turn), f'{start}-{end}')
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        # Skip if already done
        if os.path.exists(os.path.join(output_path, "dataset_info.json")):
            logger.info(f"Turn {turn} exists, skipping")
            continue

        # ====== Load or construct query data ======
        try:
            dataset = datasets.load_from_disk(cache_dir)
            logger.info(f"Reusing query cache: {cache_dir}")
        except Exception:
            if turn == 1:
                dataset = datasets.load_from_disk(args.dataset_path).select(range(start, end))
            else:
                last_turn_path = os.path.join(args.output_dir, model_tag, str(turn - 1), f'{start}-{end}')
                dataset = datasets.load_from_disk(last_turn_path)

            logger.info(f"Building query messages for {len(dataset)} instances...")

            # Build query messages for each instance
            all_messages = []
            all_selected_did = []
            for i in range(len(dataset)):
                msgs, sel_did = build_query_messages(dataset[i])
                all_messages.append(msgs)
                all_selected_did.append(sel_did)

            # --- API batch query generation ---
            # For the prefix-continuation approach, we use a simulated prompt:
            # We concatenate user prompt + assistant prefix into a single user message
            # since OpenAI API doesn't support continue_final_message natively.
            api_messages_list = []
            for msgs in all_messages:
                # msgs[0] = user prompt, msgs[1] = assistant prefix with content "**Question:**\n["
                # Combine into a single-turn prompt
                user_content = msgs[0]['content']
                # Add instruction that the answer should start with [doc_id]
                api_messages_list.append([
                    {'role': 'user', 'content': user_content},
                    {'role': 'assistant', 'content': '**Question:**\n[', 'prefix': True},
                ])

            logger.info(f"Sending {len(api_messages_list)} query requests to API...")
            results = query_gen.generate_sync(
                api_messages_list, sampling_params,
                desc=f"Turn {turn}/{max_turn} query",
            )

            # Parse results
            parsed_rows = []
            valid_indices = []
            for i, (result, sel_did) in enumerate(zip(results, all_selected_did)):
                if result is None:
                    continue
                raw_text = result['content'] or ''
                existing_messages = dataset[i].get('messages', None)
                parsed = parse_query_output(raw_text, sel_did, dataset[i]['ctxs'], existing_messages)
                if parsed is not None:
                    # Carry over original fields
                    row = dict(dataset[i])
                    row.update(parsed)
                    row['completion_tokens'] = result['completion_tokens']
                    row.pop('selected_did', None)
                    row.pop('input_ids', None)
                    row.pop('raw_question', None)
                    parsed_rows.append(row)
                    valid_indices.append(i)

            logger.info(f"Valid queries: {len(parsed_rows)}/{len(dataset)} "
                        f"({len(parsed_rows)/len(dataset)*100:.1f}%)")

            if not parsed_rows:
                logger.error("No valid queries generated! Skipping this turn.")
                continue

            dataset = datasets.Dataset.from_list(parsed_rows)
            dataset.save_to_disk(cache_dir)
            logger.info(f"Query cache saved: {cache_dir}")

        # ====== Response phase ======
        try:
            existing = datasets.load_from_disk(output_path)
            logger.info(f"Turn {turn} response already done, skipping")
            continue
        except Exception:
            pass

        logger.info(f"Building response messages for {len(dataset)} instances...")
        response_messages_list = []
        for i in range(len(dataset)):
            msgs = build_response_messages(dataset[i])
            response_messages_list.append(msgs)

        logger.info(f"Sending {len(response_messages_list)} response requests to API...")
        resp_results = response_gen.generate_sync(
            response_messages_list, sampling_params,
            desc=f"Turn {turn}/{max_turn} response",
        )

        # Merge responses into dataset
        completion_tokens_list = []
        formatted_responses = []
        for i, result in enumerate(resp_results):
            if result is None:
                completion_tokens_list.append(0)
                formatted_responses.append({'role': 'assistant', 'content': '[ERROR: generation failed]'})
            else:
                content = strip_thinking(result['content'] or '')
                completion_tokens_list.append(result['completion_tokens'])
                formatted_responses.append({'role': 'assistant', 'content': content})

        # Add completion tokens
        existing_tokens = list(dataset['completion_tokens'])
        merged_tokens = [a + b for a, b in zip(existing_tokens, completion_tokens_list)]
        dataset = safe_add_column(dataset, 'completion_tokens', merged_tokens)

        # Append response messages
        messages_col = list(dataset['messages'])
        for i, resp in enumerate(formatted_responses):
            messages_col[i] = messages_col[i] + [resp]
        dataset = safe_add_column(dataset, 'messages', messages_col)

        # Remove temporary columns
        for col in ['input_ids', 'query']:
            if col in dataset.column_names:
                dataset = dataset.remove_columns([col])

        dataset.save_to_disk(output_path)
        logger.info(f"Turn {turn} complete → {output_path}")

    logger.info("All turns generation complete!")
