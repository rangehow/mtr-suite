# Centralized Configuration Guide / 集中配置指南

> **English** sections come first; **中文** sections follow each topic with the heading `（中文）`.

---

## Table of Contents / 目录

1. [Environment Variables / 环境变量](#1-environment-variables--环境变量)
2. [Model Registry (`models.conf`) / 模型注册表](#2-model-registry-modelsconf--模型注册表)
3. [Per-Module Argument Parsers (`arg_parser.py`) / 各模块参数解析器](#3-per-module-argument-parsers-arg_parserpy--各模块参数解析器)
4. [Shared Utilities (`shared/`) / 共享工具模块](#4-shared-utilities-shared--共享工具模块)
5. [Hardware & Software Prerequisites / 硬件与软件前提条件](#5-hardware--software-prerequisites--硬件与软件前提条件)

---

## 1. Environment Variables / 环境变量

All shell scripts reference three **shell-level** environment variables. Set them before running any script, or accept the defaults shown:

| Variable | Default | Meaning |
|---|---|---|
| `MTR_ROOT` | `/path/to/mtr` | Root of this repository |
| `HOME_DIR` | `/path/to` | Your home or workspace directory (where `.bashrc`, `datasets/` live) |
| `DATASETS_DIR` | `/path/to/datasets` | Root directory of external datasets (Wikipedia dumps, ShareGPT, etc.) |
| `BASE_DIR` | `/path/to` | Base prefix for some eval scripts (resolves `$BASE_DIR/env/.bashrc`) |

**Example — add to your `~/.bashrc`:**

```bash
export MTR_ROOT="/home/user/mtr-suite"
export HOME_DIR="/home/user"
export DATASETS_DIR="/home/user/datasets"
export BASE_DIR="/home/user"
```

### （中文）

所有 shell 脚本通过三个 **shell 级别的环境变量** 来定位路径。请在运行任何脚本之前设置它们：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `MTR_ROOT` | `/path/to/mtr` | 本仓库根目录 |
| `HOME_DIR` | `/path/to` | 你的主目录或工作空间目录（`.bashrc`、`datasets/` 所在处） |
| `DATASETS_DIR` | `/path/to/datasets` | 外部数据集的根目录（Wikipedia dumps、ShareGPT 等） |
| `BASE_DIR` | `/path/to` | 部分评测脚本使用的基础前缀 |

**示例 — 添加至 `~/.bashrc`：**

```bash
export MTR_ROOT="/home/user/mtr-suite"
export HOME_DIR="/home/user"
export DATASETS_DIR="/home/user/datasets"
export BASE_DIR="/home/user"
```

---

## 2. Model Registry (`models.conf`) / 模型注册表

Several modules use a **`models.conf`** file (INI-like, `name=path` format) to decouple model names from filesystem paths. This avoids hard-coding paths into shell scripts.

**Locations (4 copies, one per module):**

| File | Used by |
|---|---|
| `generate/models.conf` | `generate/main.sh`, `generate/run.sh` |
| `rewrite/models.conf` | `rewrite/main.sh` |
| `model_choice/models.conf` | `model_choice/main.sh` |
| `analysis_of_previous_benchmark/models.conf` | `analysis_of_previous_benchmark/main.sh` |

**Format:**

```ini
# models.conf
# Format: model_name=model_path
Qwen2.5-72B-Instruct=/path/to/models/Qwen/Qwen2.5-72B-Instruct
Mistral-Large-Instruct-2411=/path/to/models/mistralai/Mistral-Large-Instruct-2411
GLM-4-32B-0414=/path/to/models/THUDM/GLM-4-32B-0414
```

Shell scripts look up model paths at runtime via a helper function:

```bash
get_model_path() {
  local name="$1" file="$2"
  local line=$(grep "^${name}=" "$file" | head -n 1)
  [ -n "$line" ] && echo "${line#*=}" || echo ""
}
query_model_path=$(get_model_path "Qwen2.5-72B-Instruct" "models.conf")
```

### （中文）

多个模块使用 **`models.conf`** 文件（INI 格式，`name=path`）将模型名称与文件系统路径解耦，避免在 shell 脚本中硬编码路径。

**位置（共 4 份，每个模块一份）：**

| 文件 | 使用者 |
|---|---|
| `generate/models.conf` | `generate/main.sh`, `generate/run.sh` |
| `rewrite/models.conf` | `rewrite/main.sh` |
| `model_choice/models.conf` | `model_choice/main.sh` |
| `analysis_of_previous_benchmark/models.conf` | `analysis_of_previous_benchmark/main.sh` |

**格式：**

```ini
# models.conf
# 格式: 模型名称=模型路径
Qwen2.5-72B-Instruct=/path/to/models/Qwen/Qwen2.5-72B-Instruct
```

Shell 脚本在运行时通过辅助函数查找模型路径：

```bash
query_model_path=$(get_model_path "Qwen2.5-72B-Instruct" "models.conf")
```

---

## 3. Per-Module Argument Parsers (`arg_parser.py`) / 各模块参数解析器

Each major module defines its own `arg_parser.py` with an `argparse`-based `parse_args()` function. The table below summarizes all arguments per module.

### `data_process/arg_parser.py`

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--dataset_path` | str | ✅ | — | Path to the raw corpus (HuggingFace dataset or local path) |
| `--save_dir` | str | ✅ | — | Output directory for processed data |
| `--max_length` | int | — | 2048 | Maximum chunk length for document splitting |
| `--quality_model_path` | str | — | — | Path to NVIDIA quality classifier (DeBERTa) |
| `--fineweb_model_path` | str | — | — | Path to NVIDIA FineWeb education classifier |

### `embedding/arg_parser.py`

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--embedding_model` | str | — | — | Embedding model name or path |
| `--processed_dataset_path` | str | — | — | Path to the processed dataset |
| `--embedding_output_dir` | str | — | — | Directory to save embedding vectors |
| `--index_output_dir` | str | — | — | Directory to save FAISS index |
| `--model_name` | str | — | — | Short model name identifier |
| `--topk` | int | ✅ | — | Top-K value for clustering neighbor search |
| `--query_batch_size` | int | — | 2097152 | Batch size for queries (too large → OOM) |
| `--cluster_dataset_output_dir` | str | ✅ | — | Output directory for clustered dataset |
| `--cluster_size` | int | ✅ | — | Number of documents per cluster |

### `generate/arg_parser.py`

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--start` | str | — | — | Start index of data range to process |
| `--end` | str | — | — | End index of data range to process |
| `--output_dir` | str | — | — | Output directory for generated dialogues |
| `--dataset_path` | str | — | — | Path to clustered dataset |
| `--query_model_name` | str | — | — | Name of the query generation model |
| `--query_model_path` | str | — | — | Path to the query generation model |
| `--response_model_name` | str | — | — | Name of the response generation model |
| `--response_model_path` | str | — | — | Path to the response generation model |
| `--turn` | str | — | — | Current dialogue turn number |
| `--cache_dir` | str | — | — | Cache directory for intermediate results |
| `--inference_backend` | str | — | — | Backend engine: `vllm` or `sglang` |
| `--last_turn_dataset` | str | — | — | Path to previous turn's output dataset |

### `model_choice/arg_parser.py`

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--start` | int | — | — | Start index |
| `--end` | int | — | — | End index |
| `--input_dir` | str | — | — | Input directory with generated data |
| `--output_dir` | str | — | — | Output directory for scored data |
| `--model_path` | str | — | — | Path to judge model |
| `--turn` | str | — | — | Turn number to evaluate |
| `--inference_backend` | str | — | — | Backend: `vllm` or `sglang` |
| `--tested_model_name` | str | — | — | Name of the model being evaluated |
| `--judge_model_name` | str | — | — | Name of the judge model |

### `analysis_of_previous_benchmark/arg_parser.py`

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--chatrag-bench-dir` | str | — | — | Path to ChatRAG Bench data |
| `--coral-dir` | str | — | — | Path to CORAL data |
| `--target` | str | — | — | What to judge: `query` or `response` |
| `--output_dir` | str | — | — | Output directory |
| `--model_path` | str | — | — | Judge model path |
| `--turn` | str | — | — | Turn number |
| `--inference_backend` | str | — | — | Backend: `vllm` or `sglang` |
| `--judge_model_name` | str | — | — | Judge model name |

### `statistic/arg_parser.py`

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--model` | str | — | — | LLM for domain classification |
| `--processed_dataset_path` | str | — | — | Processed dataset path |
| `--sharegpt_path` | str | — | — | ShareGPT dataset path |
| `--domain_classifier_path` | str | — | — | Domain classifier model path |
| `--domain_result_path` | str | — | — | Output path for domain results |
| `--coral_document_dir` | str | — | — | CORAL document corpus path |
| `--coral_dataset_dir` | str | — | — | CORAL conversation dataset path |
| `--mtr_test_path` | str | — | — | MTR test dataset path |
| `--mtr_train_path` | str | — | — | MTR train dataset path |

### `train/arg_parser.py`

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--loss-backend` | str | — | `triton` | Loss computation backend: `pytorch` or `triton` |

> **Note:** `train/main.sh` passes most training arguments directly on the command line to `train.py`, which uses its own inline `argparse` — separate from `train/arg_parser.py`.

### `eval/main.py` (inline argparse)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--embedding_output_path` | str | — | — | Root directory for embedding files |
| `--embedding_model_name` | str | — | — | Embedding model name |
| `--domain_map_path` | str | — | — | Comma-separated domain map JSON paths (or `full`) |
| `--dataset_name` | str | — | — | Comma-separated dataset names |
| `--index_output_path` | str | — | — | FAISS index directory |
| `--query_dataset_path` | str | — | — | Comma-separated query dataset paths |
| `--results_save_dir` | str | — | — | Directory to save evaluation results |

### （中文）

每个主要模块各自定义 `arg_parser.py`，使用 `argparse` 构建 `parse_args()` 函数。上表汇总了各模块所有参数。

关键约定：
- 每个子阶段的主入口是 `main.py`
- 对应的启动示例在 `main.sh`
- 参数定义在 `arg_parser.py`（`eval/` 例外，参数内联定义在 `main.py` 中）
- `train/arg_parser.py` 仅定义 `--loss-backend`，其余训练参数通过 `train.py` 内部 argparse 定义

---

## 4. Shared Utilities (`shared/`) / 共享工具模块

The `shared/` package centralizes functions that were previously duplicated across modules:

| Module | Exported Functions | Used By |
|---|---|---|
| `shared/data_utils.py` | `parse_dataset()`, `parse_dataset_with_truncation()` | `embedding/`, `eval/`, `data_process/` |
| `shared/map_utils.py` | `merge_by_add()`, `merge_by_replace()`, `merge_by_append()` | `generate/`, `model_choice/`, `analysis_of_previous_benchmark/` |
| `shared/llm_utils.py` | `initialize_llm()`, `run_generate()`, `extract_text_and_tokens()`, `shutdown_vllm()` | `generate/`, `rewrite/`, `model_choice/`, `analysis_of_previous_benchmark/` |
| `shared/embedding_utils.py` | `split_sentences()`, `embed_text()`, `run_parallel_embeddings()` | `embedding/`, `eval/` |
| `shared/faiss_utils.py` | `make_vres_vdev()`, `move_index_to_gpu()` | `embedding/`, `eval/` |
| `shared/judge_prompts.py` | All judge prompt templates (`JUDGE_QUERYER`, etc.) | `generate/`, `model_choice/`, `analysis_of_previous_benchmark/` |

**Import pattern:**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.data_utils import parse_dataset
from shared.llm_utils import initialize_llm, run_generate, extract_text_and_tokens
```

### （中文）

`shared/` 包集中了之前在各模块中重复出现的函数：

| 模块 | 导出函数 | 使用者 |
|---|---|---|
| `shared/data_utils.py` | `parse_dataset()`, `parse_dataset_with_truncation()` | `embedding/`, `eval/`, `data_process/` |
| `shared/map_utils.py` | `merge_by_add()`, `merge_by_replace()`, `merge_by_append()` | `generate/`, `model_choice/`, `analysis_of_previous_benchmark/` |
| `shared/llm_utils.py` | `initialize_llm()`, `run_generate()`, `extract_text_and_tokens()`, `shutdown_vllm()` | `generate/`, `rewrite/`, `model_choice/`, `analysis_of_previous_benchmark/` |
| `shared/embedding_utils.py` | `split_sentences()`, `embed_text()`, `run_parallel_embeddings()` | `embedding/`, `eval/` |
| `shared/faiss_utils.py` | `make_vres_vdev()`, `move_index_to_gpu()` | `embedding/`, `eval/` |
| `shared/judge_prompts.py` | 所有 judge 提示模板 | `generate/`, `model_choice/`, `analysis_of_previous_benchmark/` |

**引用方式：**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.data_utils import parse_dataset
```

---

## 5. Hardware & Software Prerequisites / 硬件与软件前提条件

### Minimum Hardware

| Resource | Requirement | Notes |
|---|---|---|
| GPU VRAM | 6 × 80 GB (e.g. A100/H100) | For LLM inference with TP=4 on 70B+ models |
| System RAM | ~600 GB | TopiOCQA evaluation alone requires ~350 GB for embeddings |
| Disk | ~500 GB free | For embeddings, indices, generated datasets |

### Software

| Dependency | Version | Notes |
|---|---|---|
| Python | ≥3.10 | |
| PyTorch | ≥2.0 | CUDA support required |
| Transformers | ≥4.40 | For ModernBERT, Qwen models |
| FAISS | `faiss-gpu` or `faiss-cpu` | ⚠️ Requires `numpy<2` (dependency conflict) |
| Inference Backend | `vllm>=0.6` **or** `sglang>=0.4` | SGLang does not support GLM4 as of v0.4.5 |
| flash-attn | Latest | Training only |

### Conda Environment Setup

```bash
# Create environment
conda create -n mtr python=3.10
conda activate mtr
pip install -r requirements.txt

# For FAISS (may need a separate env due to numpy<2)
conda create -n faiss python=3.10
conda activate faiss
pip install faiss-gpu numpy==1.26.4
```

### （中文）

### 最低硬件要求

| 资源 | 需求 | 备注 |
|---|---|---|
| GPU 显存 | 6 × 80 GB（如 A100/H100） | 用于 70B+ 模型 TP=4 推理 |
| 系统内存 | ~600 GB | 仅 TopiOCQA 评测就需要 ~350 GB 用于嵌入向量 |
| 磁盘 | ~500 GB 空闲 | 用于嵌入向量、索引、生成的数据集 |

### 软件依赖

| 依赖 | 版本 | 备注 |
|---|---|---|
| Python | ≥3.10 | |
| PyTorch | ≥2.0 | 需 CUDA 支持 |
| Transformers | ≥4.40 | 用于 ModernBERT、Qwen 模型 |
| FAISS | `faiss-gpu` 或 `faiss-cpu` | ⚠️ 需要 `numpy<2`（存在依赖冲突） |
| 推理后端 | `vllm>=0.6` **或** `sglang>=0.4` | SGLang v0.4.5 不支持 GLM4 |
| flash-attn | 最新版 | 仅训练时需要 |

### Conda 环境设置

```bash
# 创建环境
conda create -n mtr python=3.10
conda activate mtr
pip install -r requirements.txt

# FAISS 可能需要独立环境（因 numpy<2 的依赖冲突）
conda create -n faiss python=3.10
conda activate faiss
pip install faiss-gpu numpy==1.26.4
```

---
