# Quickstart: Synthesizing an MTR Dataset with Remote Qwen3.5-FP8 Nodes

This guide walks through the complete pipeline from **raw corpus → finished MTR training dataset**, using 5 remote sglang-deployed Qwen3.5-FP8 nodes as the generation backend.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 0: data_process/  (local CPU — no GPU needed)                       │
│    Raw Wikipedia/corpus → filter → quality score → chunk split             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stage 1: embedding/     (local GPU — 1-2 GPUs for embedding + FAISS)     │
│    Chunks → embed → FAISS index → K-Means cluster                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stage 2: generate/main_api.py  (API calls to remote sglang nodes) ★ NEW  │
│    Clusters → Turn 1 Q&A → Turn 2 Q&A → ... → Turn N Q&A → Rewrite       │
│    Uses 4× Qwen3.5-FP8 nodes via OpenAI-compatible API                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stage 3: train/         (local GPU)                                       │
│    MTR data → fine-tune bi-encoder retrieval model                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stage 4: eval/          (local GPU)                                       │
│    Document embed → FAISS index → query embed → retrieval → metrics        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

```bash
cd /path/to/mtr-suite
pip install -r requirements.txt
pip install aiohttp   # for API backend

export MTR_ROOT="$(pwd)"
```

### Verify Endpoints

```bash
for ip in <node1-ip> <node2-ip> <node3-ip>; do
    echo -n "$ip: "
    curl -s --max-time 5 "http://${ip}:8080/v1/models" | python3 -c \
        "import sys,json; print('OK -', json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "DOWN"
done
```

Expected: all nodes reporting `OK - <your-model-id>`

---

## Stage 0: Corpus Preprocessing

You need a raw text corpus. The default is English Wikipedia:

```bash
# Option A: Download Wikipedia (takes ~30 min)
python -c "
from datasets import load_dataset
ds = load_dataset('NeuML/wikipedia-20250123', split='train')
ds.save_to_disk('${MTR_ROOT}/mtr-data-dumps/raw_corpus')
"

# Option B: Use any HuggingFace dataset with a 'text' column
# e.g. for finance domain:
# python -c "from datasets import load_dataset; load_dataset('...',split='train').save_to_disk('mtr-data-dumps/raw_corpus')"
```

Then preprocess:

```bash
QUALITY_MODEL="nvidia/quality-classifier-deberta"
FINEWEB_MODEL="nvidia/nemocurator-fineweb-nemotron-4-edu-classifier"

python data_process/main.py \
    --dataset_path mtr-data-dumps/raw_corpus \
    --save_dir mtr-data-dumps/processed_dataset \
    --max_length 2048 \
    --quality_model_path "$QUALITY_MODEL" \
    --fineweb_model_path "$FINEWEB_MODEL"
```

**Output:** `mtr-data-dumps/processed_dataset/` — cleaned, chunked documents.

---

## Stage 1: Embedding & Clustering

```bash
# Use Qwen3-Embedding-4B (available on the model platform)
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-4B"

python embedding/main.py \
    --embedding_model "$EMBEDDING_MODEL" \
    --model_name qwen3-emb-4b \
    --processed_dataset_path mtr-data-dumps/processed_dataset \
    --embedding_output_dir tempfile/embedding \
    --topk 100 \
    --index_output_dir tempfile/index \
    --cluster_dataset_output_dir mtr-data-dumps/cluster_dataset \
    --cluster_size 8
```

**Output:** `mtr-data-dumps/cluster_dataset/` — document clusters (8 docs each), ready for generation.

---

## Stage 2: Dialogue Generation (API Backend) ★

This is the main step. Uses `generate/main_api.py` and `shared/api_utils.py` to call your remote Qwen3.5-FP8 nodes.

### Quick test (10 samples, 1 turn)

```bash
python generate/main_api.py \
    --dataset_path mtr-data-dumps/cluster_dataset \
    --query_endpoints "http://<node1>:8080/v1/chat/completions,http://<node2>:8080/v1/chat/completions" \
    --model_id qwen35-fp8 \
    --model_display_name Qwen3.5-FP8 \
    --output_dir mtr-data-dumps/mtr \
    --start 0 --end 10 --turn 1 \
    --cache_dir tempfile/cache \
    --max_concurrent 20 \
    --enable_thinking false
```

### Full generation (all data, 4 turns)

Edit and run the batch script:

```bash
# Review/edit the configuration section at the top:
vim generate/run_api.sh

# Then launch:
bash generate/run_api.sh
```

Key configuration in `run_api.sh`:

| Parameter | Default | Description |
|---|---|---|
| `ENDPOINTS` | 4 working nodes | Comma-separated API URLs |
| `MODEL_ID` | `qwen35-fp8` | As reported by sglang `/v1/models` |
| `TURN` | 4 | Number of dialogue turns (paper uses 4→7+) |
| `GLOBAL_END` | 5999 | Adjust to your cluster dataset size |
| `TASK_LENGTH` | 1000 | Samples per parallel task |
| `MAX_CONCURRENT` | 80 | Total in-flight API requests |
| `ENABLE_THINKING` | false | Set `true` for reasoning (slower, more tokens) |

### What happens during generation

For each turn, the script:
1. **Query phase:** Sends document clusters → LLM generates a question targeting one specific document
2. **Parse & filter:** Extracts `[doc_id] question` format, filters invalid responses
3. **Response phase:** Sends the question + ground truth document → LLM generates an answer
4. **Save:** Arrow dataset with messages, ground truth doc, completion tokens

### Conversational Rewriting (Stage 3 of generation)

After multi-turn generation, rewrite explicit queries into natural conversational style:

```bash
# TODO: Add phrase_api.py for API-based rewriting, or use the existing phrase.py with local model
# For now, the multi-turn data can be used directly for training
```

### Merge & Split

```bash
# Merge parallel shards
python generate/merge_result.py \
    --dataset_dir mtr-data-dumps/mtr/Qwen3.5-FP8-Qwen3.5-FP8 \
    --output_dir mtr-data-dumps/mtr_merged

# Train/test split
python generate/split.py \
    --dataset_path mtr-data-dumps/mtr_merged \
    --output_dir mtr-data-dumps
```

**Output:** `mtr-data-dumps/mtr_train/` and `mtr-data-dumps/mtr_test/`

---

## Stage 3: Training

```bash
torchrun --nproc-per-node 4 train/train.py \
    --train_dataset_name mtr \
    --save_name mtr_qwen35_modernbert_base \
    --model_dir /path/to/gte-modernbert-base \
    --mtr_dataset mtr-data-dumps/mtr_train \
    --output_dir tempfile/model
```

---

## Stage 4: Evaluation

```bash
bash eval/main.sh
```

---

## New Files Added

| File | Purpose |
|---|---|
| `shared/api_utils.py` | Async API client with round-robin load balancing |
| `generate/main_api.py` | Generation script using remote API endpoints |
| `generate/run_api.sh` | Batch execution script for API-based generation |
| `QUICKSTART_API.md` | This guide |

---

## Tips

- **Throughput:** With 4 nodes × ~20 req/s each ≈ 80 req/s. For 6000 clusters × 4 turns × 2 phases = ~48K API calls → ~10 minutes.
- **Thinking mode:** `enable_thinking=false` is recommended for data synthesis (faster, cheaper). Qwen3.5 reasoning is overkill for generating questions.
- **Fault tolerance:** Dead nodes are automatically retried on other endpoints. The script resumes from the last completed turn on restart.
- **Cost:** Roughly 8K tokens per turn per sample (prompt + completion). 6000 × 4 turns × 8K ≈ 192M tokens total.
- **Scaling:** Increase `MAX_CONCURRENT` if nodes are underutilized. Check sglang metrics at `http://<ip>:8080/metrics`.
