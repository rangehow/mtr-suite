"""
Merge shards + apply hard topic switch at a RANDOM turn + train/test split.

Hard topic switch: two conversations (A, B) from different clusters are spliced.
  - Turns 1..K come from conversation A
  - Turns K+1..12 come from conversation B (but prefixed with A's history)
  This simulates a user suddenly changing topic mid-conversation.

Old behavior: always switch at the midpoint (turn 4 of 8).
New behavior: switch turn K is sampled uniformly from [2, total_turns-1] per pair.

Usage:
    python generate/merge_with_topic_switch.py \
        --data_dir mtr-data-dumps/mtr/Qwen3.5-FP8-Qwen3.5-FP8 \
        --output_dir mtr-data-dumps/mtr_final \
        --total_turns 12 \
        --test_size 1000 \
        --seed 42
"""

import argparse
import os
import random
from collections import defaultdict

import datasets
from loguru import logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True,
                   help='e.g. mtr-data-dumps/mtr/Qwen3.5-FP8-Qwen3.5-FP8')
    p.add_argument('--output_dir', required=True,
                   help='Where to save mtr_train / mtr_test')
    p.add_argument('--total_turns', type=int, default=12)
    p.add_argument('--test_size', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--switch_min', type=int, default=2,
                   help='Earliest turn at which topic can switch (inclusive, 1-based)')
    p.add_argument('--switch_max', type=int, default=None,
                   help='Latest turn at which topic can switch (inclusive). Default: total_turns - 1')
    return p.parse_args()


def load_all_turns(data_dir, total_turns):
    """Load per-turn datasets.
    
    Returns: dict[turn_num] -> list of rows (each row is a dict)
    
    turn_num is 1-based.  Each row at turn T has T*2 messages and the 
    ground_truth for turn T.
    """
    turn_data = {}
    for t in range(1, total_turns + 1):
        turn_dir = os.path.join(data_dir, str(t))
        if not os.path.isdir(turn_dir):
            raise FileNotFoundError(f"Missing turn directory: {turn_dir}")

        shards = sorted(
            [d for d in os.listdir(turn_dir)
             if os.path.isdir(os.path.join(turn_dir, d)) and '-' in d],
            key=lambda x: int(x.split('-')[0])
        )

        all_rows = []
        for shard in shards:
            shard_path = os.path.join(turn_dir, shard)
            if not os.path.exists(os.path.join(shard_path, "dataset_info.json")):
                logger.warning(f"Skipping incomplete: {shard_path}")
                continue
            ds = datasets.load_from_disk(shard_path)
            all_rows.extend([ds[i] for i in range(len(ds))])

        turn_data[t] = all_rows
        logger.info(f"  Turn {t:>2}: {len(all_rows)} rows, "
                     f"{len(all_rows[0]['messages'])} msgs/row")

    # Sanity: all turns should have same row count
    counts = {t: len(rows) for t, rows in turn_data.items()}
    if len(set(counts.values())) != 1:
        logger.warning(f"Inconsistent row counts across turns: {counts}")
        # Trim all turns to the minimum count so indices align
        min_count = min(counts.values())
        for t in turn_data:
            turn_data[t] = turn_data[t][:min_count]
        logger.info(f"Trimmed all turns to {min_count} rows")

    return turn_data


def build_switched_conversations(turn_data, total_turns, switch_min, switch_max, seed):
    """Pair up conversations and splice at random turn.
    
    For each pair (A, B):
      - Pick switch_turn K ~ Uniform[switch_min, switch_max]
      - Output conversation 1: A's turns 1..K, then B's turns K+1..12
        (B's turns are prefixed with A's first K turns of messages as history)
      - Output conversation 2: B's turns 1..K, then A's turns K+1..12
    
    This doubles the dataset since each pair produces 2 conversations.
    But we start by pairing N rows → N/2 pairs → N conversations.
    
    Returns list of dicts with keys: messages, ctxs, ground_truth_document_idx, 
    ground_truth_ctx, switch_turn, cluster_idx_a, cluster_idx_b
    """
    rng = random.Random(seed)
    n = len(turn_data[1])
    
    # Shuffle indices and pair them up
    indices = list(range(n))
    rng.shuffle(indices)
    
    # Make pairs (drop last if odd)
    pairs = []
    for i in range(0, len(indices) - 1, 2):
        pairs.append((indices[i], indices[i + 1]))
    
    logger.info(f"Created {len(pairs)} pairs from {n} conversations")
    logger.info(f"Switch turn range: [{switch_min}, {switch_max}]")
    
    final_data = []
    switch_turn_dist = defaultdict(int)
    
    for idx_a, idx_b in pairs:
        # Random switch turn for this pair
        switch_turn = rng.randint(switch_min, switch_max)
        switch_turn_dist[switch_turn] += 1
        
        # === Conversation 1: A's front + B's back ===
        # A's messages from turns 1..switch_turn (get from turn_data[switch_turn])
        a_front_messages = turn_data[switch_turn][idx_a]['messages']  # 2*switch_turn msgs
        
        # B's messages from turns switch_turn+1..total_turns
        # B's full conversation is at turn_data[total_turns][idx_b]['messages'] (24 msgs)
        b_full_messages = turn_data[total_turns][idx_b]['messages']
        b_back_messages = b_full_messages[switch_turn * 2:]  # msgs after switch point
        
        # Splice: A's history + B's remaining turns
        conv1_messages = list(a_front_messages) + list(b_back_messages)
        
        # Ground truth: for the LAST turn, it comes from B
        # ctxs: use B's ctxs (since the later questions are about B's cluster)
        conv1 = {
            'messages': conv1_messages,
            'ctxs': turn_data[total_turns][idx_b]['ctxs'],
            'ground_truth_document_idx': turn_data[total_turns][idx_b]['ground_truth_document_idx'],
            'ground_truth_ctx': turn_data[total_turns][idx_b]['ground_truth_ctx'],
            'switch_turn': switch_turn,
            'cluster_idx_a': turn_data[1][idx_a]['cluster_idx'],
            'cluster_idx_b': turn_data[1][idx_b]['cluster_idx'],
        }
        
        # === Conversation 2: B's front + A's back ===
        b_front_messages = turn_data[switch_turn][idx_b]['messages']
        a_full_messages = turn_data[total_turns][idx_a]['messages']
        a_back_messages = a_full_messages[switch_turn * 2:]
        
        conv2_messages = list(b_front_messages) + list(a_back_messages)
        
        conv2 = {
            'messages': conv2_messages,
            'ctxs': turn_data[total_turns][idx_a]['ctxs'],
            'ground_truth_document_idx': turn_data[total_turns][idx_a]['ground_truth_document_idx'],
            'ground_truth_ctx': turn_data[total_turns][idx_a]['ground_truth_ctx'],
            'switch_turn': switch_turn,
            'cluster_idx_a': turn_data[1][idx_b]['cluster_idx'],
            'cluster_idx_b': turn_data[1][idx_a]['cluster_idx'],
        }
        
        final_data.append(conv1)
        final_data.append(conv2)
    
    # Log switch turn distribution
    logger.info("Switch turn distribution:")
    for t in sorted(switch_turn_dist):
        bar = "█" * (switch_turn_dist[t] * 40 // max(switch_turn_dist.values()))
        logger.info(f"  Turn {t:>2}: {switch_turn_dist[t]:>5} pairs  {bar}")
    
    return final_data


def train_test_split(dataset, test_size, segment_len=8):
    """Spread-out test split."""
    total = len(dataset)
    if test_size >= total:
        test_size = max(1, total // 10)
    
    num_segments = test_size // segment_len
    max_start = total - segment_len
    
    if max_start < 0 or num_segments <= 0:
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
    
    return dataset.select(train_idx), dataset.select(test_idx)


if __name__ == "__main__":
    args = parse_args()
    
    if args.switch_max is None:
        args.switch_max = args.total_turns - 1
    
    random.seed(args.seed)
    
    # === Step 1: Load all turns ===
    logger.info("=" * 60)
    logger.info("Step 1: Loading all turn data")
    logger.info("=" * 60)
    turn_data = load_all_turns(args.data_dir, args.total_turns)
    
    # === Step 2: Build topic-switched conversations ===
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Hard topic switch (random turn)")
    logger.info("=" * 60)
    switched = build_switched_conversations(
        turn_data, args.total_turns,
        args.switch_min, args.switch_max, args.seed,
    )
    logger.info(f"Total conversations after switch: {len(switched)}")
    
    # Verify message count
    msg_lens = [len(c['messages']) for c in switched]
    expected = args.total_turns * 2
    assert all(l == expected for l in msg_lens), \
        f"Expected {expected} messages/row, got {set(msg_lens)}"
    logger.info(f"All conversations have {expected} messages ✓")
    
    # Show a sample
    sample = switched[0]
    logger.info(f"\nSample (switch at turn {sample['switch_turn']}):")
    logger.info(f"  Cluster A: {sample['cluster_idx_a']}, Cluster B: {sample['cluster_idx_b']}")
    for i, msg in enumerate(sample['messages']):
        if msg['role'] == 'user':
            turn_num = i // 2 + 1
            marker = " ← SWITCH" if turn_num == sample['switch_turn'] + 1 else ""
            logger.info(f"  Turn {turn_num:>2}: {msg['content'][:80]}...{marker}")
    
    # === Step 3: Create dataset and split ===
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Train/test split")
    logger.info("=" * 60)
    
    dataset = datasets.Dataset.from_list(switched)
    dataset = dataset.shuffle(seed=args.seed)
    
    train_ds, test_ds = train_test_split(dataset, args.test_size)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, 'mtr_train')
    test_path = os.path.join(args.output_dir, 'mtr_test')
    
    train_ds.save_to_disk(train_path)
    test_ds.save_to_disk(test_path)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"✅ Done!")
    logger.info(f"  Train: {train_path} ({len(train_ds)} rows)")
    logger.info(f"  Test:  {test_path} ({len(test_ds)} rows)")
    logger.info(f"  Turns: {args.total_turns}, Messages/row: {args.total_turns * 2}")
    logger.info(f"  Switch range: turn {args.switch_min}~{args.switch_max} (uniform random)")
    logger.info(f"{'=' * 60}")
