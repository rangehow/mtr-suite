<div align="center">

# MTR-Suite

**一个用于评估和合成对话检索基准的框架**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ACL 2026](https://img.shields.io/badge/ACL-2026%20Main-red.svg)](https://2026.aclweb.org/)
[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-blue.svg)](https://huggingface.co/datasets/OkayestProgrammer/mtr-qwen35-fp8-12turn)
[![Paper](https://img.shields.io/badge/📄%20Paper-Coming%20Soon-green.svg)]()

[English](readme.md) | [中文](readme_zh.md)

</div>

---

本仓库包含 **MTR-Suite: A Framework for Evaluating and Synthesizing Conversational Retrieval Benchmarks**（ACL 2026 Main）的代码。

MTR-Suite 是一个用于**审计**、**合成**和**评测**对话检索系统的统一框架，包含：
- **MTR-Eval**：基于 LLM 的审计器，量化已有基准中的标注偏差。
- **MTR-Pipeline**：基于贪心遍历聚类的多智能体系统，以人工标注 1/400 的成本生成高质量对话。
- **MTR-Bench**：严格的通用领域基准，通过真实对话现象（硬主题切换、冗长回复、长上下文歧义）压力测试现代检索器。

## 文档

我们提供三份详细指南文档（中英双语）：

| 文档 | 说明 |
|---|---|
| **[`CONFIGURATION.md`](CONFIGURATION.md)** | 集中配置参考：环境变量、`models.conf` 模型注册表、各模块参数表、`shared/` 共享工具、硬件与软件前提条件。 |
| **[`EXECUTION.md`](EXECUTION.md)** | 分步执行指南：快速开始、端到端流水线（阶段 0–4）、所有辅助模块、完整 Shell 脚本索引。 |
| **[`CONTRIBUTING.md`](CONTRIBUTING.md)** | 贡献指南：项目约定、代码风格、如何添加新模块/后端/数据集/指标、常见陷阱、PR 检查清单。 |

## 安装

```bash
git clone https://github.com/rangehow/mtr-suite.git
cd mtr-suite
pip install -r requirements.txt
```

**注意：** 部分依赖根据使用场景可选。详见 `requirements.txt`：
- **推理后端：** 根据偏好安装 `vllm` 或 `sglang`（或两者均安装）。
- **FAISS：** GPU 加速检索安装 `faiss-gpu`，仅 CPU 安装 `faiss-cpu`。
- **训练：** `flash-attn` 和 `triton` 仅在训练时需要。

## 协议

本项目所有资源以 **MIT 协议** 开源，具体参见 `LICENSE` 文件。

## 项目结构

我们尽量保证代码在易读性和拓展性上达到平衡。为此，每个单独的功能部分拆分成独立的文件夹，每个功能模块都可以剥离本项目独立运行。

**每个文件夹内都有一个更细致的 `readme.md`**，帮助开发者更好地理解、使用或修改该功能模块。每个可单独执行的脚本（实现了 `if __name__ == '__main__'`）都有一个对应的 `.sh` 示例文件。

### 目录概览

* `data_process/` — 原始数据清洗、过滤及切块
* `embedding/` — 文档嵌入、FAISS GPU 索引、贪心遍历聚类
* `generate/` — 数据生成（查询、回复、对话改写）
* `train/` — 三元组数据上的训练（anchor, positive, negatives）
* `eval/` — 实验评测，包含嵌入-索引-评测子流程
* `model_choice/` — 模型选型与打分
* `analysis_of_previous_benchmark/` — 对 ChatRAG 涉及的 5 个数据集做细粒度打分
* `statistic/` — 数据集统计、领域分类、话题流分析
* `data_labeling/` — 人工标注工具与标注者一致性分析
* `ablation_study/` — 论文中的消融实验
* `sundries/` — 杂项工具（上传 HuggingFace、处理历史数据集等）
* `shared/` — 共享工具模块（数据加载、LLM 后端、嵌入工具、FAISS 工具、提示模板等）

## 主要生产流水线

```
data_process/ → embedding/ → generate/
```

每个子阶段的主入口是 `main.py`，启动示例在 `main.sh`，参数定义在 `arg_parser.py`。

详细流程请参见 [EXECUTION.md](EXECUTION.md)。

## 局限性

* **SGLang 推理后端：** 我们**不建议**使用 SGLang（v0.4.5）完全复现我们的流水线，因为部分模型（如 GLM4）尚未被该版本支持。但如果是合成你自己的基准数据集，可以使用任何你熟悉的工具。

## API 访问

本项目支持三种推理后端：
1. **vLLM** — 本地 GPU 推理
2. **SGLang** — 本地 GPU 推理
3. **OpenAI 兼容 API** — 远程端点（如 sglang 部署的模型）

API 后端详见 [`QUICKSTART_API.md`](QUICKSTART_API.md) 和 `generate/main_api.py`，支持异步批量请求、多节点轮询负载均衡和自动重试。

## 硬件需求

完成本项目所有实验至少需要：
* **6 张 80GB 显存的 GPU**
* 约 **600GB 系统内存**

（服务器内存通常配置为显存两倍以上，因此主要关注总显存大小即可。）

## 训练

目前仓库里的训练脚本比较基础，仅支持 shared encoder（query 和 document 的 embedding 模型共享参数）。如需更鲁棒的训练过程，推荐参考：
* **Dual Encoder 训练：** [facebookresearch/DPR](https://github.com/facebookresearch/DPR)
* **LLM-based Encoder 训练：** [GTE-Qwen](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#fine-tuning)，依赖 `modelscope/ms-swift`
* 基于 [sentence-transformers](https://github.com/UKPLab/sentence-transformers) 训练也是可行的

## 资源

### 数据集

| 资源 | 说明 |
|---|---|
| [**MTR-Qwen3.5-FP8-12Turn**](https://huggingface.co/datasets/OkayestProgrammer/mtr-qwen35-fp8-12turn) | 🌟 **推荐使用。** 最新最高质量版本。12 轮对话 + 随机硬主题切换，由 Qwen3.5-FP8 (397B MoE) 生成。**使用 `train` split 训练，`test` split 作为 MTR-Bench 评测。** |
| [**MTR-DOCUMENT**](https://huggingface.co/datasets/OkayestProgrammer/MTR-DOCUMENT) | 用于检索的文档集合（1,041,047 篇维基百科段落，来自 2025-01 dump）。 |

> **关于论文版本与开源版本的说明：** 论文中报告的结果使用的是早期模型组合生成的 8 轮版本。我们后续升级了流水线，发布了更高质量的 12 轮版本（由 Qwen3.5-FP8 生成）作为推荐基准。用户可自行从 `train` 集划分 train/dev；建议使用 `test` split 进行评测。

### 原始数据集（论文中使用）

这些是论文中引用的原始数据集。建议新工作使用上述 Qwen3.5-FP8-12Turn 版本。

| 资源 | 说明 |
|---|---|
| [**MTR-BENCH**](https://huggingface.co/datasets/OkayestProgrammer/MTR-BENCH) | 论文原始测试集（8 轮）。 |
| [**MTR-TRAIN**](https://huggingface.co/datasets/OkayestProgrammer/MTR-train) | 论文原始训练集（8 轮）。 |

## 引用

如果您觉得本工作有帮助，请引用我们的论文：

```bibtex
@inproceedings{ruan-etal-2026-mtr-suite,
    title={{MTR}-Suite: A Framework for Evaluating and Synthesizing Conversational Retrieval Benchmarks},
    author={Junhao Ruan and Abudukeyumu Abudula and Bei Li and Yongjing Yin and Xinyu Liu and Kechen Jiao and Xin Chen and Jingang Wang and Xunliang Cai and Tong Xiao and Jingbo Zhu},
    booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
    year={2026}
}
```

## 贡献

我们欢迎贡献！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。如有任何关于使用或扩展代码的问题，欢迎：
* 提交 Issue（bug、问题、功能请求）
* 提交 Pull Request

感谢您的关注与合作！
