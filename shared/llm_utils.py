"""
Shared LLM backend utilities for initializing, running inference, and extracting results.
Supports both vLLM and SGLang backends.
"""
import contextlib
import gc
import os
from loguru import logger


def shutdown_vllm(llm):
    """Gracefully shut down a vLLM engine and free GPU resources."""
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment,
    )
    import ray
    import torch
    
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()


def run_generate(llm, dataset, sampling_params, tokenizer, backend):
    """Run generation on a dataset using the specified backend.
    
    Args:
        llm: The LLM engine (vLLM or SGLang).
        dataset: Dataset with 'input_ids' column.
        sampling_params: Backend-specific sampling parameters.
        tokenizer: The tokenizer for decoding (used by vLLM).
        backend: 'vllm' or 'sglang'.
        
    Returns:
        Raw outputs from the backend.
    """
    if backend == 'sglang':
        outputs = llm.generate(input_ids=dataset['input_ids'], sampling_params=sampling_params)
    elif backend == 'vllm':
        # prompt_token_ids is deprecated since v0.10
        input_text = tokenizer.batch_decode(dataset['input_ids'])
        outputs = llm.generate(input_text, sampling_params=sampling_params)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    logger.info("Generation process done.")
    return outputs


def initialize_llm(model_name, model_path, backend, context_length=32768):
    """Initialize an LLM engine.
    
    Args:
        model_name: Name of the model (used to determine TP size).
        model_path: Path to model weights.
        backend: 'vllm' or 'sglang'.
        context_length: Maximum context length. Default 32768.
        
    Returns:
        Initialized LLM engine.
    """
    from torch.cuda import device_count
    
    if model_name in {'GLM-4-32B-0414', 'Gemma-3-27b-it'}:
        tp_size = 2
    else:
        tp_size = 4
    
    dp_size = device_count() // tp_size
    
    if backend == 'sglang':
        import sglang as sgl
        llm = sgl.Engine(
            model_path=model_path,
            tp_size=tp_size,
            dp_size=dp_size,
            context_length=context_length,
            show_time_cost=True,
            log_requests=True,
            enable_metrics=True,
            decode_log_interval=1,
        )
    elif backend == 'vllm':
        from vllm import LLM
        llm = LLM(
            model=model_path,
            tensor_parallel_size=device_count(),
            enable_prefix_caching=True,
            task='generate',
            max_model_len=context_length,
            dtype='bfloat16',
            max_num_seqs=64,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    logger.info(f"{backend} engine initialized: {model_name}")
    return llm


def extract_text_and_tokens(outputs, tokenizer, backend):
    """Extract output text and completion token counts from backend outputs.
    
    Args:
        outputs: Raw outputs from the backend.
        tokenizer: The tokenizer for token counting.
        backend: 'vllm' or 'sglang'.
        
    Returns:
        Tuple of (output_text_list, completion_tokens_list).
    """
    if backend == 'sglang':
        output_text = [item['text'] for item in outputs]
        completion_tokens = [item['meta_info']['completion_tokens'] for item in outputs]
    elif backend == 'vllm':
        output_text = [item.outputs[0].text.strip() for item in outputs]
        completion_tokens = tokenizer(output_text, return_length=True).length
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    return output_text, completion_tokens
