<div align="center">

# MTR-Suite

**A Framework for Evaluating and Synthesizing Conversational Retrieval Benchmarks**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ACL 2026](https://img.shields.io/badge/ACL-2026%20Main-red.svg)](https://2026.aclweb.org/)
[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Datasets-blue.svg)](https://huggingface.co/datasets/OkayestProgrammer/mtr-qwen35-fp8-12turn)
[![Paper](https://img.shields.io/badge/📄%20Paper-Coming%20Soon-green.svg)]()

[English](readme.md) | [中文](readme_zh.md)

</div>

---

This repository contains the codebase for **MTR-Suite: A Framework for Evaluating and Synthesizing Conversational Retrieval Benchmarks** (ACL 2026 Main).

MTR-Suite is a unified framework for **auditing**, **synthesizing**, and **benchmarking** conversational retrieval systems. It features:
- **MTR-Eval**: An LLM-based auditor quantifying alignment gaps in previous benchmarks.
- **MTR-Pipeline**: A multi-agent system using greedy traversal clustering to generate high-fidelity dialogues at 1/400th human cost.
- **MTR-Bench**: A rigorous general-domain benchmark that stress-tests modern retrievers with realistic conversational phenomena (hard topic switching, verbosity, long-context ambiguity).

## Table of Contents

*   [Installation](#installation)
*   [License](#license)
*   [Documentation](#documentation)
*   [Project Structure](#project-structure)
    *   [Directory Overview](#directory-overview)
*   [Main Production Pipeline](#main-production-pipeline)
*   [Limitations](#limitations)
*   [API Access](#api-access)
*   [Hardware Requirements](#hardware-requirements)
*   [Training](#training)
*   [Resources](#resources)
*   [Citation](#citation)
*   [Contributing](#contributing)

## Installation

```bash
git clone https://github.com/rangehow/mtr-suite.git
cd mtr-suite
pip install -r requirements.txt
```

**Note:** Some dependencies are optional depending on your use case. See `requirements.txt` for details. In particular:
- **Inference backends:** Install either `vllm` or `sglang` (or both) depending on your preference.
- **FAISS:** Install `faiss-gpu` for GPU-accelerated retrieval, or `faiss-cpu` for CPU-only.
- **Training:** `flash-attn` and `triton` are only needed if you plan to train models.

## License

All resources in this project are open-sourced under the **MIT License**. Please see the `LICENSE` file for more details.

## Documentation

We provide three detailed guide documents (bilingual English/Chinese):

| Document | Description |
|---|---|
| **[`CONFIGURATION.md`](CONFIGURATION.md)** | Centralized configuration reference: environment variables, `models.conf` registry, per-module argument tables, `shared/` utilities, hardware/software prerequisites. |
| **[`EXECUTION.md`](EXECUTION.md)** | Step-by-step execution guide: quick start, end-to-end pipeline walkthrough (Stage 0–4), all auxiliary modules, and a complete shell script index. |
| **[`CONTRIBUTING.md`](CONTRIBUTING.md)** | Contributing guidelines: project conventions, code style, how to add new modules/backends/datasets/metrics, common pitfalls, and PR checklist. |

## Project Structure

We strive for a balance between code readability and extensibility. To achieve this, individual functional components are organized into separate directories. Each module is designed to be self-contained and can be run independently of the main project.

**Crucially, each directory contains its own detailed `readme.md` file** to help developers better understand, use, or modify that specific functional module.

Furthermore, every executable script (i.e., those implementing `if __name__ == '__main__':`) has a corresponding example `.sh` file to demonstrate its usage.

### Directory Overview

*   `analysis_of_previous_benchmark/`
    *   Provides fine-grained scoring for the 5 datasets involved in ChatRAG.
*   `data_process/`
    *   Handles raw data cleaning, mixing, filtering, and chunking.
*   `embedding/`
    *   Implements efficient clustering, including document embedding, FAISS indexing on GPU, and greedy clustering.
*   `eval/`
    *   Manages the evaluation of our experiments, including the embedding-indexing-evaluation sub-pipeline.
*   `generate/`
    *   Responsible for data generation (queries, responses, and conversational rephrasing).
*   `model_choice/`
    *   Manages model selection. This module is functionally similar to `analysis_of_previous_benchmark/` and could potentially be merged. However, they are currently separate for convenience due to differences in dataset structures.
*   `train/`
    *   Facilitates training on triplet data (anchor, positive, negatives). Note: Currently supports a single positive example and multiple optional negative examples.
*   `statistic/`
    *   Dataset statistics, domain classification, and topic flow analysis.
*   `sundries/`
    *   Contains miscellaneous utility scripts, such as uploading datasets to Hugging Face and processing various past benchmark datasets.
*   `data_labeling/`
    *   Human annotation tools and inter-annotator agreement analysis.
*   `ablation_study/`
    *   Scripts for ablation experiments reported in the paper.

## Main Production Pipeline

The main production pipeline, corresponding to the repository structure, is as follows:

`data_process/` → `embedding/` → `generate/`

The primary entry point for each sub-stage is `main.py`. Corresponding example startup scripts can be found in `main.sh`, and parameters are defined in `arg_parser.py` within each respective module's directory.

## Limitations

*   **SGLang for Inference:** We do **not** recommend using SGLang (version 0.4.5) as the inference backend if you aim to completely reproduce our pipeline. This is because some models we utilized (e.g., GLM4) are not yet supported by this version of SGLang. However, for synthesizing your own benchmarks, feel free to use any tool you are familiar with that works well with your chosen models.

## API Access

This project supports three inference backends:
1. **vLLM** — local GPU inference
2. **SGLang** — local GPU inference
3. **OpenAI-compatible API** — remote endpoints (e.g., sglang served models)

For the API backend, see [`QUICKSTART_API.md`](QUICKSTART_API.md) and `generate/main_api.py`. It supports async batching, round-robin load balancing across multiple nodes, and automatic retries.

## Hardware Requirements

Please note: To complete all experiments in this project, you will need at least:
*   **6 GPUs with 80GB of VRAM each**
*   Approximately **600GB of system RAM**
    *(Considering that server RAM is typically configured to be at least twice the total VRAM, your primary concern should be ensuring sufficient total VRAM.)*

## Training

The current training scripts in this repository are relatively basic. They only support a **shared encoder** (where query and document embedding models share parameters) and do not support:
*   **Dual encoders** (e.g., similar to DRAGON+)
*   **LLM-based encoders** (e.g., GTE-Qwen)

For more robust and customizable training processes, we recommend referring to the following repositories:
*   **Dual Encoder Training:** See [facebookresearch/DPR](https://github.com/facebookresearch/DPR).
*   **LLM-based Encoder Training:** Refer to the GTE-Qwen training methodology ([Alibaba-NLP/gte-Qwen2-7B-instruct#fine-tuning](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#fine-tuning)), which relies on the external library `modelscope/ms-swift`.
*   Training using [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) is also a viable option.

## Resources

### Datasets

| Resource | Description |
|---|---|
| [**MTR-Qwen3.5-FP8-12Turn**](https://huggingface.co/datasets/OkayestProgrammer/mtr-qwen35-fp8-12turn) | 🌟 **Recommended.** Our latest and highest-quality dataset. 12-turn conversations with random hard topic switches, generated by Qwen3.5-FP8 (397B MoE). **Use the `train` split for training and `test` split as MTR-Bench.** |
| [**MTR-DOCUMENT**](https://huggingface.co/datasets/OkayestProgrammer/MTR-DOCUMENT) | Document collection for retrieval (1,041,047 Wikipedia passages from the 2025-01 dump). |

> **Note on paper vs. open-source version:** The paper reports results using an 8-turn version generated by an earlier model combination. We have since upgraded the pipeline and are releasing a higher-quality 12-turn version (generated by Qwen3.5-FP8) as the recommended benchmark. Users can split the `train` set into train/dev as needed; we recommend the `test` split for benchmarking.

### Legacy Datasets (from the paper)

These are the original datasets referenced in the paper. We recommend using the Qwen3.5-FP8-12Turn version above for new work.

| Resource | Description |
|---|---|
| [**MTR-BENCH**](https://huggingface.co/datasets/OkayestProgrammer/MTR-BENCH) | Original benchmark test set from the paper (8-turn). |
| [**MTR-TRAIN**](https://huggingface.co/datasets/OkayestProgrammer/MTR-train) | Original training set from the paper (8-turn). |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ruan-etal-2026-mtr-suite,
    title={{MTR}-Suite: A Framework for Evaluating and Synthesizing Conversational Retrieval Benchmarks},
    author={Junhao Ruan and Abudukeyumu Abudula and Bei Li and Yongjing Yin and Xinyu Liu and Kechen Jiao and Xin Chen and Jingang Wang and Xunliang Cai and Tong Xiao and Jingbo Zhu},
    booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
    year={2026}
}
```

## Contributing

We welcome contributions! If you have any questions about using or extending the code, or if you'd like to contribute, please feel free to:
*   Submit an Issue for bugs, questions, or feature requests.
*   Open a Pull Request with your proposed changes.

We appreciate your interest and collaboration!
