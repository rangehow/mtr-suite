# Contributing Guidelines / 贡献指南

> **English** sections come first; **中文** sections follow each topic with the heading `（中文）`.

---

## Table of Contents / 目录

1. [Project Conventions / 项目约定](#1-project-conventions--项目约定)
2. [Code Style / 代码风格](#2-code-style--代码风格)
3. [Adding a New Module / 添加新模块](#3-adding-a-new-module--添加新模块)
4. [Adding a New Inference Backend / 添加新推理后端](#4-adding-a-new-inference-backend--添加新推理后端)
5. [Adding a New Dataset / 添加新数据集](#5-adding-a-new-dataset--添加新数据集)
6. [Adding a New Evaluation Metric / 添加新评测指标](#6-adding-a-new-evaluation-metric--添加新评测指标)
7. [Common Pitfalls / 常见陷阱](#7-common-pitfalls--常见陷阱)
8. [Pull Request Checklist / PR 检查清单](#8-pull-request-checklist--pr-检查清单)

---

## 1. Project Conventions / 项目约定

### Directory Structure

Each functional module is an **independent directory** that can run standalone. Every module follows a consistent layout:

```
module_name/
├── main.py           # Primary entry point
├── main.sh           # Example shell script showing how to invoke main.py
├── arg_parser.py     # argparse-based CLI arguments
├── readme.md         # Module-specific documentation
├── models.conf       # (if applicable) Model name → path registry
└── *.py / *.sh       # Additional scripts
```

### Key Rules

1. **Entry point = `main.py`** — The primary entry point for each module is always `main.py`.
2. **Example script = `main.sh`** — Every executable script (`if __name__ == '__main__'`) has a corresponding `.sh` file.
3. **Arguments = `arg_parser.py`** — CLI arguments are defined in `arg_parser.py` (exceptions: `eval/main.py` and `train/train.py` define args inline).
4. **Shared code lives in `shared/`** — Duplicate logic must be extracted to `shared/`. Do NOT copy-paste functions between modules.
5. **No hardcoded paths** — Use `${MTR_ROOT}`, `${HOME_DIR}`, `${DATASETS_DIR}` in `.sh` files. Use argparse or config files in `.py` files.
6. **No debug artifacts** — No `breakpoint()`, `pdb.set_trace()`, or `import pdb` in committed code.
7. **English only** — All comments, docstrings, log messages, and documentation must be in English (README may have bilingual versions as `readme.md` + `readme_zh.md`).
8. **Logging** — Use `loguru.logger` for structured logging. Avoid bare `print()` for status messages.

### File Naming

| Pattern | Convention |
|---|---|
| `a.py`, `b.py`, `c.py` | Scratch/experimental files — gitignored |
| `*.hope` | Cluster job templates — gitignored |
| `*.jsonl`, `*.json`, `*.csv` | Data files — gitignored |
| `_foo.py` | Internal sub-module (e.g., `embedding/_embedding.py`) |
| `foo_bar.py` | Public module |

### （中文）

### 目录结构

每个功能模块是一个**独立目录**，可以脱离项目独立运行。每个模块遵循一致的布局：

```
module_name/
├── main.py           # 主入口
├── main.sh           # 启动示例
├── arg_parser.py     # CLI 参数定义
├── readme.md         # 模块文档
├── models.conf       # 模型注册表（如适用）
└── *.py / *.sh       # 其他脚本
```

### 核心规则

1. **入口 = `main.py`**
2. **示例 = `main.sh`** — 每个可执行脚本都有对应的 `.sh` 文件
3. **参数 = `arg_parser.py`**
4. **共享代码放 `shared/`** — 禁止在模块间复制粘贴函数
5. **禁止硬编码路径** — `.sh` 中使用 `${MTR_ROOT}` 等变量，`.py` 中使用 argparse 或配置文件
6. **禁止调试代码** — 禁止提交 `breakpoint()`、`pdb.set_trace()`
7. **仅限英文** — 所有注释、文档字符串、日志信息均使用英文
8. **日志** — 使用 `loguru.logger`，避免用 `print()` 输出状态信息

---

## 2. Code Style / 代码风格

### Python

- **Formatter:** No enforced formatter (yet). Use consistent 4-space indentation.
- **Type hints:** Encouraged but not required.
- **Docstrings:** Use Google-style docstrings for public functions.
- **Imports:** Standard library → third-party → `shared.*` → local modules.

```python
# Good
import os
import numpy as np
from loguru import logger

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.data_utils import parse_dataset

from arg_parser import parse_args
```

### Shell

- Use `${VAR:-default}` for all path variables.
- Use `set -e` at the top of production scripts.
- Quote all variable expansions: `"$var"`, not `$var`.

### （中文）

### Python

- **缩进：** 4 空格
- **类型提示：** 鼓励但不强制
- **文档字符串：** 公共函数使用 Google 风格
- **导入顺序：** 标准库 → 第三方 → `shared.*` → 本地模块

### Shell

- 所有路径变量使用 `${VAR:-default}` 格式
- 生产脚本顶部使用 `set -e`
- 变量展开加引号：`"$var"`

---

## 3. Adding a New Module / 添加新模块

1. Create a new directory: `my_module/`
2. Add `main.py` with `if __name__ == '__main__':`
3. Add `arg_parser.py` with `parse_args()` function
4. Add `main.sh` with example invocation
5. Add `readme.md` describing the module
6. If you need shared utilities, import from `shared/` — do NOT duplicate
7. Update `readme.md` (root) to add the module to the Directory Overview

### （中文）

1. 创建新目录：`my_module/`
2. 添加 `main.py`，包含 `if __name__ == '__main__':`
3. 添加 `arg_parser.py`，包含 `parse_args()` 函数
4. 添加 `main.sh` 启动示例
5. 添加 `readme.md` 模块说明
6. 如需共享工具，从 `shared/` 导入 — 不要复制
7. 更新根目录 `readme.md`，在目录概览中添加新模块

---

## 4. Adding a New Inference Backend / 添加新推理后端

Currently supported backends: `vllm`, `sglang`.

To add a new backend (e.g., OpenAI API):

1. **Edit `shared/llm_utils.py`:**
   - Add a new branch in `initialize_llm()`, `run_generate()`, and `extract_text_and_tokens()`.

```python
# In initialize_llm():
elif backend == 'openai':
    from openai import OpenAI
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    return client

# In run_generate():
elif backend == 'openai':
    # Implement batched async requests
    ...

# In extract_text_and_tokens():
elif backend == 'openai':
    output_text = [r.choices[0].message.content for r in outputs]
    completion_tokens = [r.usage.completion_tokens for r in outputs]
    return output_text, completion_tokens
```

2. **Update `arg_parser.py`** in relevant modules to accept the new backend name.
3. **Test** with a small dataset before full-scale runs.

> See `generate/conversation_generator.py` for an existing async OpenAI implementation that could serve as a reference.

### （中文）

目前支持的推理后端：`vllm`、`sglang`。

添加新后端（如 OpenAI API）：

1. **编辑 `shared/llm_utils.py`：** 在 `initialize_llm()`、`run_generate()` 和 `extract_text_and_tokens()` 中添加新分支。
2. **更新 `arg_parser.py`** 以接受新后端名称。
3. **小规模测试** 后再进行全量运行。

> 可参考 `generate/conversation_generator.py` 中已有的异步 OpenAI 实现。

---

## 5. Adding a New Dataset / 添加新数据集

### For Evaluation

1. Prepare the dataset in one of the supported formats:
   - HuggingFace `datasets.save_to_disk()` format (preferred)
   - JSONL with fields: `messages` (list of `{role, content}`), `gold_doc_id`, `document_list`
   - TSV with `text` column (for document collection)

2. **If it's a small dataset** (per-document indexing, like Doc2Dial/QuAC):
   - Provide a `domain_map.json` mapping each conversation to its document domain
   - Add a branch in `eval/main.py`'s data loading section

3. **If it's a large dataset** (shared index, like TopiOCQA):
   - Pre-build the FAISS index
   - Use `full` as the domain_map_path placeholder

4. **Add to `eval/main.sh`** in the `dataset_configs` array:
   ```bash
   "domain_map_path;query_dataset_path;dataset_name"
   ```

### For Training

1. Format as HuggingFace dataset with columns: `history` (list of messages), `gold_idx`, `document_list`
2. Add loading logic in `train/train.py`

### （中文）

### 用于评测

1. 将数据集准备为以下格式之一：HuggingFace `save_to_disk()` 格式（推荐）、JSONL、TSV
2. **小数据集**：提供 `domain_map.json`，在 `eval/main.py` 添加加载分支
3. **大数据集**：预构建 FAISS 索引，`domain_map_path` 设为 `full`
4. 在 `eval/main.sh` 的 `dataset_configs` 数组中添加条目

### 用于训练

1. 格式化为 HuggingFace 数据集，包含 `history`、`gold_idx`、`document_list` 列
2. 在 `train/train.py` 中添加加载逻辑

---

## 6. Adding a New Evaluation Metric / 添加新评测指标

All metrics are defined in `eval/metrics.py`. Currently implemented:

- `recall(ranked_indices_list, gold_index_list, topk)` → Recall@K
- `mrr(ranked_indices_list, gold_index_list, topk)` → MRR@K
- `ndcg(ranked_indices_list, gold_index_list, topk)` → NDCG@K

To add a new metric:

1. Add the function to `eval/metrics.py` following the same signature pattern.
2. Call it in `eval/main.py` where metrics are computed.

### （中文）

所有指标定义在 `eval/metrics.py` 中。添加新指标：在该文件中添加函数，然后在 `eval/main.py` 中调用。

---

## 7. Common Pitfalls / 常见陷阱

| Issue | Cause | Solution |
|---|---|---|
| FAISS crashes with `numpy>=2` | FAISS has `numpy<2` dependency | Use a separate conda env for FAISS operations |
| Dragon+ scores drop with infinity backend | infinity defaults to mean pooling; Dragon+ needs CLS pooling | Use `sentence-transformers` backend for Dragon+ |
| TopiOCQA eval hangs | Not enough GPU VRAM for index; `move_index_to_gpu` blocks | Set `ngpus=0` in code to use CPU index |
| SGLang doesn't support GLM4 | SGLang v0.4.5 limitation | Use vLLM for GLM4 |
| Multi-turn queries repeat | LLM ignores "don't repeat" instruction | Anti-repetition fix: explicit `{PREVIOUS_QUESTIONS}` list in prompt (already implemented) |
| `flash_attn` import error | Missing flash-attn package | Only needed for training; install with `pip install flash-attn` |
| Import error for `shared.*` | `sys.path` not configured | Add `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))` before imports |

### （中文）

| 问题 | 原因 | 解决方案 |
|---|---|---|
| FAISS 在 `numpy>=2` 下崩溃 | FAISS 依赖 `numpy<2` | 为 FAISS 操作使用独立 conda 环境 |
| Dragon+ 分数下降（使用 infinity 后端） | infinity 默认 mean pooling，Dragon+ 需要 CLS pooling | 对 Dragon+ 使用 `sentence-transformers` 后端 |
| TopiOCQA 评测卡死 | GPU 显存不足以加载索引 | 将代码中 `ngpus` 设为 0，使用 CPU 索引 |
| SGLang 不支持 GLM4 | SGLang v0.4.5 限制 | 对 GLM4 使用 vLLM |
| 多轮查询重复 | LLM 忽略"不要重复"指令 | 已实现反重复修复：在提示中显式列出 `{PREVIOUS_QUESTIONS}` |
| `flash_attn` 导入错误 | 未安装 flash-attn | 仅训练时需要：`pip install flash-attn` |
| `shared.*` 导入错误 | `sys.path` 未配置 | 在导入前添加 `sys.path.insert(0, ...)` |

---

## 8. Pull Request Checklist / PR 检查清单

Before submitting a PR, verify:

- [ ] No `breakpoint()`, `pdb.set_trace()`, or `import pdb`
- [ ] No hardcoded absolute paths (use env variables or argparse)
- [ ] All comments and docstrings in English
- [ ] New scripts have corresponding `.sh` examples
- [ ] Shared logic uses `shared/` — no copy-paste duplication
- [ ] `requirements.txt` updated if new dependencies introduced
- [ ] `readme.md` updated if new modules or major features added
- [ ] Tested with at least one dataset end-to-end

### （中文）

提交 PR 前请确认：

- [ ] 无 `breakpoint()`、`pdb.set_trace()` 或 `import pdb`
- [ ] 无硬编码绝对路径
- [ ] 所有注释和文档字符串均为英文
- [ ] 新脚本有对应的 `.sh` 示例
- [ ] 共享逻辑使用 `shared/` — 无复制粘贴
- [ ] 如引入新依赖，已更新 `requirements.txt`
- [ ] 如添加新模块或重要功能，已更新 `readme.md`
- [ ] 至少在一个数据集上完成端到端测试

---
