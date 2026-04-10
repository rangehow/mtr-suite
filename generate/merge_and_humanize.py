"""
Post-processing pipeline for MTR dataset:
  1. Merge shards across all turns into a single dataset
  2. Humanize user queries (make them sound like real human speech)
  3. Train/test split

Usage:
    python generate/merge_and_humanize.py \
        --data_dir mtr-data-dumps/mtr/Qwen3.5-FP8-Qwen3.5-FP8 \
        --output_dir mtr-data-dumps/mtr_final \
        --endpoints "http://<node1>:8080/v1/chat/completions,..." \
        --model_id qwen35-fp8 \
        --max_concurrent 200 \
        --test_size 1000
"""

import argparse
import copy
import os
import random
import re
import sys

import datasets
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.api_utils import ApiBatchGenerator


HUMANIZE_PROMPT = """You're helping simulate how a real person chats. Take the last user message and rewrite it the way someone would ACTUALLY type it — brief, lazy, maybe grammatically rough.

Real humans in conversations:
- Ask short questions: "what about the name change?" not "What name change did this lodge undergo in 2015?"  
- Use "that", "it", "they" instead of repeating full names
- Skip obvious context: "and the ceremony?" not "What about the Ordeal Ceremony?"
- Sometimes just ask one thing, not two compound questions
- Don't capitalize perfectly or use fancy words
- Might write "whats" instead of "what's", "didnt" instead of "didn't"

If the query asks about 2+ things, pick the more interesting one or compress both into <12 words.

Return ONLY the rewritten query. No explanation.

{CONVERSATION}
[Original]: {QUERY}

[Human version]:"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True,
                   help='Path to generated data, e.g. mtr-data-dumps/mtr/Qwen3.5-FP8-Qwen3.5-FP8')
    p.add_argument('--output_dir', required=True,
                   help='Where to save final train/test datasets')
    p.add_argument('--endpoints', required=True,
                   help='Comma-separated API endpoints for humanization')
    p.add_argument('--model_id', default='qwen35-fp8')
    p.add_argument('--max_concurrent', type=int, default=200)
    p.add_argument('--test_size', type=int, default=1000,
                   help='Number of test samples (spread across dataset)')
    p.add_argument('--skip_humanize', action='store_true',
                   help='Skip humanization, just merge and split')
    p.add_argument('--skip_first_turn', action='store_true', default=True,
                   help='Do not humanize turn 1 queries (they have no context to compress)')
    p.add_argument('--humanize_cache', default='tempfile/cache/humanize_cache',
                   help='Cache dir for humanization results')
    return p.parse_args()


def load_and_merge_shards(data_dir):
    """Merge all shards across all turns into a unified dataset.
    
    Structure: data_dir/{turn}/{start-end}/  (HuggingFace datasets)
    
    Returns a single dataset where each row has the full multi-turn conversation.
    The last turn's data contains the complete messages list.
    """
    # Discover turns
    turns = sorted([int(d) for d in os.listdir(data_dir) if d.isdigit()])
    if not turns:
        raise ValueError(f"No turn directories found in {data_dir}")
    max_turn = max(turns)
    logger.info(f"Found {len(turns)} turns: {turns[0]}..{max_turn}")

    # The last turn has the complete conversation (each turn appends to messages)
    last_turn_dir = os.path.join(data_dir, str(max_turn))
    shard_dirs = sorted(os.listdir(last_turn_dir), key=lambda x: int(x.split('-')[0]))

    all_datasets = []
    for shard in shard_dirs:
        shard_path = os.path.join(last_turn_dir, shard)
        if not os.path.exists(os.path.join(shard_path, "dataset_info.json")):
            logger.warning(f"Skipping incomplete shard: {shard_path}")
            continue
        ds = datasets.load_from_disk(shard_path)
        all_datasets.append(ds)
        logger.info(f"  Loaded {shard}: {len(ds)} rows")

    merged = datasets.concatenate_datasets(all_datasets)
    logger.info(f"Merged dataset: {len(merged)} rows, {len(merged[0]['messages'])} messages each")
    return merged


def build_humanize_messages(conversation_messages, turn_idx):
    """Build API messages for humanizing a specific user turn.
    
    turn_idx: 0-based index of the user turn to humanize (turn 0 = first Q&A pair)
    """
    user_msg_idx = turn_idx * 2  # user messages are at even indices
    
    if user_msg_idx >= len(conversation_messages):
        return None
    
    # Build conversation context (everything before this turn)
    context_parts = []
    for msg in conversation_messages[:user_msg_idx]:
        role = "User" if msg['role'] == 'user' else "Assistant"
        # Truncate long assistant responses to save tokens
        content = msg['content']
        if msg['role'] == 'assistant' and len(content) > 300:
            content = content[:300] + "..."
        context_parts.append(f"[{role}]: {content}")
    
    conversation_str = "\n".join(context_parts) if context_parts else "(Start of conversation)"
    query = conversation_messages[user_msg_idx]['content']
    
    prompt = HUMANIZE_PROMPT.format(CONVERSATION=conversation_str, QUERY=query)
    
    return [{'role': 'user', 'content': prompt}]


def humanize_dataset(dataset, api_gen, skip_first_turn=True, cache_dir=None):
    """Rewrite user queries to sound more natural/human.
    
    For each conversation, rewrites turns 2+ (or all turns) user messages.
    """
    messages_list = list(dataset['messages'])
    num_turns = len(messages_list[0]) // 2  # Q&A pairs
    
    start_turn = 1 if skip_first_turn else 0
    
    logger.info(f"Humanizing {len(messages_list)} conversations × "
                f"turns {start_turn+1}..{num_turns} = "
                f"{len(messages_list) * (num_turns - start_turn)} queries")
    
    sampling_params = {
        'temperature': 0.8,
        'top_p': 0.9,
        'max_new_tokens': 100,  # Rewrites should be very short
    }
    
    # Process turn by turn to maintain consistency
    for turn_idx in range(start_turn, num_turns):
        turn_cache = os.path.join(cache_dir, f"turn_{turn_idx}") if cache_dir else None
        
        # Check cache
        if turn_cache and os.path.exists(os.path.join(turn_cache, "done")):
            logger.info(f"  Turn {turn_idx+1}: loading from cache")
            cached = datasets.load_from_disk(turn_cache)
            rewrites = list(cached['rewrite'])
            for i, rw in enumerate(rewrites):
                if rw and len(rw.strip()) > 3:
                    messages_list[i][turn_idx * 2]['content'] = rw.strip()
            continue
        
        # Build API requests
        api_msgs = []
        for i in range(len(messages_list)):
            msgs = build_humanize_messages(messages_list[i], turn_idx)
            api_msgs.append(msgs)
        
        logger.info(f"  Turn {turn_idx+1}/{num_turns}: sending {len(api_msgs)} requests...")
        results = api_gen.generate_sync(
            api_msgs, sampling_params,
            desc=f"Humanize turn {turn_idx+1}/{num_turns}",
        )
        
        # Apply rewrites
        rewrites = []
        applied = 0
        for i, result in enumerate(results):
            if result is None or not result.get('content'):
                rewrites.append(None)
                continue
            
            rewrite = result['content'].strip()
            # Strip thinking if present
            if '\n</think>\n\n' in rewrite:
                rewrite = rewrite.split('\n</think>\n\n')[-1].strip()
            
            # Basic quality checks
            original = messages_list[i][turn_idx * 2]['content']
            
            # Reject if rewrite is longer than original (should be shorter!)
            if len(rewrite) > len(original) * 1.3:
                rewrites.append(None)
                continue
            
            # Reject if too short (likely garbage)
            if len(rewrite) < 3:
                rewrites.append(None)
                continue
            
            # Reject if it contains metadata/instructions
            if any(kw in rewrite.lower() for kw in ['rewrite', 'here is', 'certainly', 'sure,']):
                rewrites.append(None)
                continue
            
            rewrites.append(rewrite)
            messages_list[i][turn_idx * 2]['content'] = rewrite
            applied += 1
        
        logger.info(f"  Turn {turn_idx+1}: applied {applied}/{len(messages_list)} rewrites "
                     f"({applied/len(messages_list)*100:.1f}%)")
        
        # Save cache
        if turn_cache:
            os.makedirs(turn_cache, exist_ok=True)
            cache_ds = datasets.Dataset.from_dict({'rewrite': [r or '' for r in rewrites]})
            cache_ds.save_to_disk(turn_cache)
            with open(os.path.join(turn_cache, "done"), 'w') as f:
                f.write("ok")
    
    # Update dataset
    dataset = dataset.remove_columns(['messages'])
    dataset = dataset.add_column('messages', messages_list)
    return dataset


def train_test_split(dataset, test_size, segment_len=8):
    """Split dataset into train/test with spread-out test segments."""
    total = len(dataset)
    
    if test_size >= total:
        logger.warning(f"test_size ({test_size}) >= dataset ({total}), using 10%")
        test_size = max(1, total // 10)
    
    # Generate spread segments (groups of 8 consecutive indices)
    num_segments = test_size // segment_len
    max_start = total - segment_len
    
    if max_start < 0 or num_segments <= 0:
        # Fallback: simple random split
        indices = list(range(total))
        random.shuffle(indices)
        test_idx = sorted(indices[:test_size])
        train_idx = sorted(indices[test_size:])
    else:
        max_block = max_start // segment_len
        test_idx = set()
        
        for i in range(num_segments):
            if num_segments == 1:
                block = 0
            else:
                frac = i * max_block / (num_segments - 1)
                block = int(frac + 0.5)
            start = block * segment_len
            for j in range(segment_len):
                if start + j < total:
                    test_idx.add(start + j)
        
        test_idx = sorted(test_idx)
        train_idx = sorted(set(range(total)) - set(test_idx))
    
    train_ds = dataset.select(train_idx)
    test_ds = dataset.select(test_idx)
    
    logger.info(f"Split: {len(train_ds)} train + {len(test_ds)} test")
    return train_ds, test_ds


if __name__ == "__main__":
    args = parse_args()
    
    # === Step 1: Merge shards ===
    logger.info("=" * 60)
    logger.info("Step 1: Merging shards")
    logger.info("=" * 60)
    merged = load_and_merge_shards(args.data_dir)
    
    # Show sample before humanization
    sample = merged[0]
    logger.info(f"\nSample BEFORE humanization:")
    for i, msg in enumerate(sample['messages']):
        if msg['role'] == 'user':
            logger.info(f"  Turn {i//2+1}: {msg['content'][:100]}")
    
    # === Step 2: Humanize queries ===
    if not args.skip_humanize:
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Humanizing user queries")
        logger.info("=" * 60)
        
        endpoints = [e.strip() for e in args.endpoints.split(',') if e.strip()]
        api_gen = ApiBatchGenerator(
            endpoints=endpoints,
            model=args.model_id,
            max_concurrent=args.max_concurrent,
            enable_thinking=False,
        )
        
        os.makedirs(args.humanize_cache, exist_ok=True)
        merged = humanize_dataset(
            merged, api_gen,
            skip_first_turn=args.skip_first_turn,
            cache_dir=args.humanize_cache,
        )
        
        # Show sample after humanization
        sample = merged[0]
        logger.info(f"\nSample AFTER humanization:")
        for i, msg in enumerate(sample['messages']):
            if msg['role'] == 'user':
                logger.info(f"  Turn {i//2+1}: {msg['content'][:100]}")
    else:
        logger.info("\nSkipping humanization (--skip_humanize)")
    
    # === Step 3: Shuffle and split ===
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Train/test split")
    logger.info("=" * 60)
    
    # Shuffle
    merged = merged.shuffle(seed=42)
    
    train_ds, test_ds = train_test_split(merged, args.test_size)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, 'mtr_train')
    test_path = os.path.join(args.output_dir, 'mtr_test')
    
    train_ds.save_to_disk(train_path)
    test_ds.save_to_disk(test_path)
    
    logger.info(f"\n✅ Done!")
    logger.info(f"  Train: {train_path} ({len(train_ds)} rows)")
    logger.info(f"  Test:  {test_path} ({len(test_ds)} rows)")
    
    # Final stats
    num_turns = len(train_ds[0]['messages']) // 2
    logger.info(f"\n  Turns per conversation: {num_turns}")
    logger.info(f"  Messages per conversation: {num_turns * 2}")
    logger.info(f"  Total train messages: {len(train_ds) * num_turns * 2:,}")
