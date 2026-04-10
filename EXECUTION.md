# Execution Guide / 执行指南

> **English** sections come first; **中文** sections follow each topic with the heading `（中文）`.

---

## Table of Contents / 目录

1. [Quick Start / 快速开始](#1-quick-start--快速开始)
2. [End-to-End Pipeline / 端到端流水线](#2-end-to-end-pipeline--端到端流水线)
3. [Stage 0: Corpus Preprocessing (`data_process/`) / 语料预处理](#3-stage-0-corpus-preprocessing-data_process--语料预处理)
4. [Stage 1: Embedding & Clustering (`embedding/`) / 嵌入与聚类](#4-stage-1-embedding--clustering-embedding--嵌入与聚类)
5. [Stage 2: Dialogue Generation (`generate/`) / 对话生成](#5-stage-2-dialogue-generation-generate--对话生成)
6. [Stage 3: Training (`train/`) / 训练](#6-stage-3-training-train--训练)
7. [Stage 4: Evaluation (`eval/`) / 评测](#7-stage-4-evaluation-eval--评测)
8. [Auxiliary Modules / 辅助模块](#8-auxiliary-modules--辅助模块)
9. [Shell Script Index / Shell 脚本索引](#9-shell-script-index--shell-脚本索引)

---

## 1. Quick Start / 快速开始

```bash
# 1. Clone and install
git clone https://github.com/rangehow/mtr-suite.git
cd mtr-suite
pip install -r requirements.txt

# 2. Set environment variables
export MTR_ROOT="$(pwd)"
export HOME_DIR="$HOME"
export DATASETS_DIR="$HOME/datasets"

# 3. Edit models.conf in generate/, rewrite/, model_choice/ to point to your model paths

# 4. Run the pipeline (each step depends on the previous one)
bash data_process/main.sh        # Stage 0: Preprocess corpus
bash embedding/main.sh           # Stage 1: Embed & cluster
bash generate/run.sh             # Stage 2: Generate multi-turn dialogues
bash train/main.sh               # Stage 3: Train retrieval model
bash eval/main.sh                # Stage 4: Evaluate
```

### （中文）

```bash
# 1. 克隆并安装
git clone https://github.com/rangehow/mtr-suite.git && cd mtr-suite
pip install -r requirements.txt

# 2. 设置环境变量
export MTR_ROOT="$(pwd)"
export HOME_DIR="$HOME"
export DATASETS_DIR="$HOME/datasets"

# 3. 编辑 generate/models.conf 等文件，将模型路径指向你的本地路径

# 4. 按顺序执行流水线（每步依赖上一步的输出）
bash data_process/main.sh        # 阶段 0：语料预处理
bash embedding/main.sh           # 阶段 1：嵌入与聚类
bash generate/run.sh             # 阶段 2：生成多轮对话
bash train/main.sh               # 阶段 3：训练检索模型
bash eval/main.sh                # 阶段 4：评测
```

---

## 2. End-to-End Pipeline / 端到端流水线

```
┌───────────────────────────────────────────────────────────────────────────┐
│ STAGE 0: data_process/                                                   │
│   Raw Corpus → filter → quality scoring → chunk splitting                │
│   Entry: data_process/main.py    Script: data_process/main.sh            │
│   Output: mtr-data-dumps/processed_dataset/                              │
├───────────────────────────────────────────────────────────────────────────┤
│ STAGE 1: embedding/                                                      │
│   Processed docs → embed → FAISS index → K-Means cluster                │
│   Entry: embedding/main.py       Script: embedding/main.sh               │
│   Output: tempfile/embedding/, tempfile/index/,                          │
│           mtr-data-dumps/cluster_dataset/                                │
├───────────────────────────────────────────────────────────────────────────┤
│ STAGE 2: generate/                                                       │
│   Clustered docs → Stage1 (1st turn) → Stage1.5 (filter)                │
│                  → Stage2 ×N (extend) → Stage1.5 (filter)                │
│                  → Stage3 (rewrite)   → pairup (combine)                 │
│   Entry: generate/main.py       Script: generate/run.sh                  │
│   Output: mtr-data-dumps/{model_name}/{turn}/{start-end}/                │
├───────────────────────────────────────────────────────────────────────────┤
│ STAGE 3: train/                                                          │
│   MTR Train Data → fine-tune bi-encoder (shared or dual)                 │
│   Entry: train/train.py          Script: train/main.sh                   │
│   Output: tempfile/model/                                                │
├───────────────────────────────────────────────────────────────────────────┤
│ STAGE 4: eval/                                                           │
│   Document embed → FAISS index → Query embed → Retrieval → Metrics      │
│   Entry: eval/main.py           Script: eval/main.sh                     │
│   Output: tempfile/eval_results/                                         │
└───────────────────────────────────────────────────────────────────────────┘
```

### （中文）

流水线按 `data_process/ → embedding/ → generate/ → train/ → eval/` 的顺序执行。每个阶段的主入口是 `main.py`，启动示例在 `main.sh`。中间结果存储在 `mtr-data-dumps/`（数据集）和 `tempfile/`（临时文件，如嵌入向量和索引）。

---

## 3. Stage 0: Corpus Preprocessing (`data_process/`) / 语料预处理

**What it does:** Loads raw Wikipedia (or other corpora), strips boilerplate sections, filters short articles, optionally applies quality/education scoring, and splits long documents into ~1024-character chunks.

**Entry point:** `data_process/main.py`

**Example:**
```bash
python data_process/main.py \
    --dataset_path /path/to/wikipedia-dump \
    --save_dir mtr-data-dumps/processed_dataset \
    --max_length 2048 \
    --quality_model_path /path/to/nvidia/quality-classifier-deberta \
    --fineweb_model_path /path/to/nvidia/fineweb-edu-classifier
```

**Key scripts in this directory:**

| Script | Purpose |
|---|---|
| `main.py` / `main.sh` | Full preprocessing pipeline |
| `filter.py` / `filter.sh` | Wikipedia filtering only |
| `minhash.py` | MinHash deduplication |
| `sort.py` | Sort/organize clustered data |

### （中文）

**功能：** 加载原始 Wikipedia（或其他语料），去除样板段落（"See also"、"References" 等），过滤短文章（<1024字符），可选进行质量/教育分类打分，并将长文档切分为约 1024 字符的块。

**入口：** `data_process/main.py`

**关键参数：**
- `--dataset_path`：原始语料路径
- `--save_dir`：处理后输出目录
- `--max_length`：最大块长度（默认 2048）
- `--quality_model_path`：NVIDIA 质量分类器路径（可选）
- `--fineweb_model_path`：NVIDIA FineWeb 教育分类器路径（可选）

---

## 4. Stage 1: Embedding & Clustering (`embedding/`) / 嵌入与聚类

**What it does:** Encodes all document chunks into dense vectors, builds a FAISS index, runs GPU-accelerated K-Means clustering, and outputs clustered document groups (each cluster = one document pool for dialogue generation).

**Entry point:** `embedding/main.py`

**Example:**
```bash
python embedding/main.py \
    --embedding_model /path/to/gte-Qwen2-7B-instruct \
    --model_name gte-qwen2 \
    --processed_dataset_path mtr-data-dumps/processed_dataset \
    --embedding_output_dir tempfile/embedding \
    --topk 100 \
    --index_output_dir tempfile/index \
    --cluster_dataset_output_dir mtr-data-dumps/cluster_dataset \
    --cluster_size 8
```

**Key sub-modules:**

| File | Purpose |
|---|---|
| `_embedding.py` | Document embedding (SentenceTransformer / infinity_emb) |
| `_indexing.py` | FAISS index building (GPU-accelerated) |
| `_clustering.py` | GPU K-Means clustering |
| `main.py` | Orchestrates all three steps |

### （中文）

**功能：** 将所有文档块编码为稠密向量，构建 FAISS 索引，运行 GPU 加速的 K-Means 聚类，输出聚类后的文档组（每个聚类 = 一个用于对话生成的文档池）。

**关键参数：**
- `--embedding_model`：嵌入模型路径
- `--cluster_size`：每个聚类的文档数量（推荐 5–8）
- `--topk`：聚类邻居搜索的 Top-K 值

---

## 5. Stage 2: Dialogue Generation (`generate/`) / 对话生成

This is the core pipeline. It has **3 sub-stages** plus a quality filter:

### Sub-stage 1: First Turn Generation

```bash
# Generate 1st turn (turn=1)
python generate/main.py \
    --dataset_path mtr-data-dumps/cluster_dataset \
    --query_model_name Qwen2.5-72B-Instruct \
    --query_model_path /path/to/Qwen2.5-72B-Instruct \
    --response_model_name Qwen2.5-72B-Instruct \
    --response_model_path /path/to/Qwen2.5-72B-Instruct \
    --output_dir mtr-data-dumps/mtr \
    --start 0 --end 999 --turn 1 \
    --cache_dir tempfile/cache \
    --inference_backend vllm
```

### Sub-stage 1.5: Quality Filtering

```bash
# Filters out responses < 120 tokens
python generate/stage1_5.py --input_path <turn_output_dir> --output_path <filtered_dir>
```

### Sub-stage 2: Multi-Turn Extension (repeat for turns 2–N)

```bash
# Extend to turn 2 (uses turn 1 output as --last_turn_dataset)
python generate/main.py \
    --turn 2 \
    --last_turn_dataset mtr-data-dumps/mtr/.../1/0-999 \
    ... # same args as above
```

### Sub-stage 3: Conversational Rewriting

```bash
# Rewrite explicit queries into natural conversational style (anaphora, ellipsis)
python generate/phrase.py \
    --dataset_path <final_turn_output> \
    --model_path /path/to/Qwen2.5-72B-Instruct \
    --inference_backend vllm
```

### Batch Execution

The `generate/run.sh` script automates the full pipeline:
- Reads `models.conf` for model paths
- Splits the dataset range into parallel tasks
- Submits jobs to the cluster scheduler (`hope run`)
- Skips already-completed ranges (checks for existing output directories)

```bash
# Edit generate/run.sh to set GLOBAL_START, GLOBAL_END, TURN, NUM_TASKS, etc.
cd generate && bash run.sh
```

### （中文）

对话生成共有 **3 个子阶段** + 质量过滤：

1. **Stage1**（`turn=1`）：为每个文档聚类生成第一轮问答
2. **Stage1.5**：过滤掉回答过短（<120 tokens）的样本
3. **Stage2**（`turn=2,3,...,N`）：迭代扩展对话到 7–10 轮
4. **Stage3**（`phrase.py`）：将显式查询改写为自然对话风格（使用指代和省略）

**批量执行：** 使用 `generate/run.sh` 自动化整个流程，支持并行任务提交和断点续跑。

**反重复机制：** 在多轮扩展（Stage2）中，提示词会列出所有已有问题，明确要求 LLM "不得重复或改述已有问题，必须针对不同的主题、方面或事实提问"。

---

## 6. Stage 3: Training (`train/`) / 训练

**What it does:** Fine-tunes a bi-encoder retrieval model on MTR triplet data (anchor=dialogue history, positive=gold document, negatives=other cluster documents).

**Entry point:** `train/train.py`

**Example:**
```bash
torchrun --nproc-per-node 4 train/train.py \
    --train_dataset_name mtr \
    --save_name mtr_modernbert_base \
    --model_dir /path/to/gte-modernbert-base \
    --chatrag_dataset /path/to/synthethisqa \
    --mtr_dataset mtr-data-dumps/mtr_train \
    --output_dir tempfile/model
```

**Key features:**
- Extends BERT position embeddings from 512 → 8192 tokens
- Left-side truncation for query encoding (preserves recent turns)
- InfoNCE contrastive loss with in-batch + hard negatives
- Supports shared encoder (`train.py`) and dual encoder (`train_share.py`)

### （中文）

**功能：** 在 MTR 三元组数据上微调双编码器检索模型。

**关键特性：**
- 将 BERT 位置嵌入从 512 扩展到 8192 tokens
- 查询编码使用左截断（保留最近的对话轮次）
- InfoNCE 对比损失，结合批内负例和硬负例
- 支持共享编码器（`train.py`）和双编码器（`train_share.py`）

---

## 7. Stage 4: Evaluation (`eval/`) / 评测

The evaluation pipeline has **3 sub-steps**: document embedding → index building → query embedding & retrieval.

### Step 1: Document Embedding

```bash
# Using sentence-transformers backend
bash eval/document_embedding_st.sh

# Or using infinity_emb backend
bash eval/document_embedding.sh
```

### Step 2: Index Building

```bash
bash eval/document_index.sh
```

### Step 3: Query Embedding & Evaluation

```bash
bash eval/main.sh
```

### All-in-one (if using the `eval/main.sh` orchestrator)

```bash
# Edit eval/main.sh to configure:
#   - embedding_model_name array (models to evaluate)
#   - dataset_configs array (datasets to evaluate on)
bash eval/main.sh
```

**Supported datasets:** Doc2Dial, QuAC, QReCC, TopiOCQA, MTR

**Metrics:** Recall@{1, 5, 20}, MRR@20, NDCG@20

### Conversational Query Rewriting (CQR) Evaluation

```bash
# First rewrite queries
bash rewrite/main.sh

# Then evaluate with rewritten queries
bash eval/rewrite_rewrite.sh
```

### Per-Turn Breakdown

```bash
bash eval/eval_for_each_turn.sh
```

### （中文）

评测流程分为 **3 个子步骤**：

1. **文档嵌入**：`eval/document_embedding_st.sh` 或 `eval/document_embedding.sh`
2. **索引构建**：`eval/document_index.sh`
3. **查询嵌入与检索**：`eval/main.sh`

**支持的数据集：** Doc2Dial, QuAC, QReCC, TopiOCQA, MTR

**指标：** Recall@{1, 5, 20}, MRR@20, NDCG@20

⚠️ **注意事项：**
- 评测 TopiOCQA 需要 ~350 GB 内存 + ~350 GB 显存
- 使用 Dragon+ 模型时不要使用 infinity 后端（默认 mean pooling 导致分数下降，Dragon+ 需要 CLS pooling）
- FAISS 依赖 `numpy<2`，可能需要独立的 conda 环境

---

## 8. Auxiliary Modules / 辅助模块

### `rewrite/` — Conversational Query Reformulation (CQR)

Rewrites the last user query in a multi-turn conversation into a standalone query.

```bash
# Rewrites with multiple models sequentially
bash rewrite/main.sh
```

### `model_choice/` — Model Selection & Scoring

Generates N×N scoring data using an LLM judge to compare different generation models.

```bash
bash model_choice/main.sh           # Generate scores
bash model_choice/response.sh       # Single-model scoring
bash model_choice/analysis_result.sh # Aggregate results into charts/CSV
```

### `analysis_of_previous_benchmark/` — ChatRAG Benchmark Analysis

Fine-grained scoring of existing ChatRAG Bench datasets (Doc2Dial, QuAC, QReCC, TopiOCQA, InSCit).

```bash
bash analysis_of_previous_benchmark/main.sh
```

### `statistic/` — Dataset Statistics

Domain classification, topic flow analysis, and dataset statistics.

```bash
bash statistic/main.sh
```

### `data_labeling/` — Human Annotation

Web-based annotation tool + inter-annotator agreement analysis.

```bash
# Launch annotation web UI
python data_labeling/annotation_web.py

# Compute correlation between human and LLM scores
python data_labeling/main.py
```

### `ablation_study/` — Ablation Experiments

Scripts for ablation studies reported in the paper:

| Subdirectory | Experiment |
|---|---|
| `health_domain/` | Domain transfer (health/finance) |
| `lmsys_chat/` | LMSYS Chat filtering |
| `mtr-eval-bias/` | Evaluation bias analysis |
| `mtr-pipeline/` | Pipeline variant comparison |
| `sharegpt/` | ShareGPT topic switch analysis |

### `sundries/` — Miscellaneous Utilities

Quick one-off scripts for data processing, format conversion, and uploads.

| Script | Purpose |
|---|---|
| `upload_dataset.py` / `upload_model.py` | Upload to HuggingFace Hub |
| `process_chatrag.py` | Process ChatRAG Bench data |
| `convert_eval_results_to_excel.py` | Aggregate eval results into Excel |
| `extract_turn_excel.py` | Extract per-turn results |
| `add_gold_idx_to_topiocqa.py` | Add gold indices to TopiOCQA |
| `draw_bar_chart.py` | Generate bar charts for paper |

### （中文）

### `rewrite/` — 对话查询改写（CQR）

将多轮对话中的最后一个用户查询改写为独立查询。

### `model_choice/` — 模型选型与打分

使用 LLM 作为评判者（judge），对不同生成模型进行 N×N 交叉打分。

### `analysis_of_previous_benchmark/` — ChatRAG 基准分析

对已有 ChatRAG Bench 数据集进行细粒度打分。

### `statistic/` — 数据集统计

领域分类、话题流分析和数据集统计。

### `data_labeling/` — 人工标注

基于 Web 的标注工具 + 标注者一致性分析。标注数据将保存到 `scores.csv`，使用 `main.py` 计算与 LLM 打分的相关性。

⚠️ 输入数据集必须来自 `model_choice/` 或 `analysis_of_previous_benchmark/` 生成的带打分数据集，否则无法读取打分理由并计算相关性系数。

### `ablation_study/` — 消融实验

论文中报告的消融实验脚本。

### `sundries/` — 杂项工具

用于数据处理、格式转换、上传至 HuggingFace 等的快脚本。开发者在复现论文时可能用到，但实际生成任务可跳过。

---

## 9. Shell Script Index / Shell 脚本索引

A complete index of all `.sh` scripts and their roles:

### Core Pipeline Scripts

| Script | Stage | Purpose |
|---|---|---|
| `data_process/main.sh` | 0 | Full corpus preprocessing |
| `data_process/filter.sh` | 0 | Wikipedia filtering only |
| `embedding/main.sh` | 1 | Embed + index + cluster |
| `generate/main.sh` | 2 | Single generation job |
| `generate/run.sh` | 2 | Batch job submission (orchestrator) |
| `generate/run_local.sh` | 2 | Local batch execution (no cluster) |
| `generate/phrase.sh` | 2 | Conversational rewriting (Stage 3) |
| `generate/merge_result.sh` | 2 | Merge parallel output shards |
| `generate/split.sh` | 2 | Split dataset for parallel processing |
| `generate/cost.sh` | 2 | Cost estimation |
| `generate/debug.sh` | 2 | Debug mode |
| `train/main.sh` | 3 | Model training |
| `eval/main.sh` | 4 | Full evaluation |
| `eval/document_embedding.sh` | 4 | Document embedding (infinity) |
| `eval/document_embedding_st.sh` | 4 | Document embedding (sentence-transformers) |
| `eval/document_index.sh` | 4 | Build FAISS indices |
| `eval/query_embedding.sh` | 4 | Query embedding (infinity) |
| `eval/query_embedding_st.sh` | 4 | Query embedding (sentence-transformers) |
| `eval/eval_for_each_turn.sh` | 4 | Per-turn evaluation |
| `rewrite/main.sh` | CQR | Query rewriting |

### CQR Evaluation Scripts

| Script | Purpose |
|---|---|
| `eval/rewrite_rewrite.sh` | Evaluate with rewritten queries |
| `eval/rewrite/query_embedding_rewrite.sh` | Embed rewritten queries |
| `eval/rewrite/query_embedding_cqr_cdr.sh` | CQR+CDR embedding |
| `eval/rewrite/query_embedding_cqr_cdr_lastturn.sh` | CQR+CDR last-turn only |
| `eval/rewrite/query_embedding_st_cqr_cdr.sh` | ST + CQR+CDR embedding |
| `eval/rewrite/query_embedding_st_cqr_cdr_lastturn.sh` | ST + CQR+CDR last-turn |

### Auxiliary Scripts

| Script | Purpose |
|---|---|
| `model_choice/main.sh` | Model selection scoring |
| `model_choice/response.sh` | Single-model response scoring |
| `model_choice/analysis_result.sh` | Aggregate model scores |
| `model_choice/hope.sh` / `local.sh` | Trial generation |
| `analysis_of_previous_benchmark/main.sh` | ChatRAG analysis |
| `analysis_of_previous_benchmark/analysis_result.sh` | Aggregate analysis |
| `statistic/main.sh` | Dataset statistics |
| `statistic/topic_flow.sh` | Topic flow analysis |
| `data_labeling/main.sh` | Human annotation pipeline |
| `data_labeling/annotation_score.sh` | Annotation scoring |
| `sundries/process_chatrag.sh` | Process ChatRAG data |
| `sundries/convert_eval_results_to_excel.sh` | Results → Excel |
| `sundries/extract_turn_excel.sh` | Per-turn results → Excel |
| `sundries/add_gold_idx_to_topiocqa.sh` | TopiOCQA gold index |
| `sundries/upload_model.sh` | Upload model to HF Hub |

### Ablation Study Scripts

| Script | Purpose |
|---|---|
| `ablation_study/health_domain/embedding.sh` | Health domain embedding |
| `ablation_study/health_domain/generate.sh` | Health domain generation |
| `ablation_study/mtr-pipeline.sh` | Pipeline ablation |
| `ablation_study/mtr-eval-bias.sh` | Eval bias ablation |
| `ablation_study/sharegpt/topic_switch.sh` | ShareGPT topic switch |
| `ablation_study/lmsys_chat/hope/run_filter.sh` | LMSYS chat filtering |

### （中文）

以上是所有 `.sh` 脚本的完整索引。核心流水线脚本按阶段 0→4 顺序排列，每个阶段的主入口都是 `main.sh`。

**约定：**
- `main.sh`：主启动脚本
- `hope.sh` / `local.sh`：用于模型选型的试运行脚本
- `run.sh` / `run_local.sh`：批量任务提交脚本
- `*_st.sh`：使用 sentence-transformers 后端的变体
- `models.conf`：模型注册表，所有 `.sh` 脚本通过它查找模型路径

---
