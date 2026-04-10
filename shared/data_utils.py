"""
Shared data loading utilities used across embedding/, eval/, and other modules.
"""
import datasets


def parse_dataset(dataset_str):
    """Parse a dataset from various formats (TSV, JSONL, or HuggingFace disk format).
    
    Args:
        dataset_str: Path to dataset file or directory.
        
    Returns:
        A HuggingFace Dataset object.
    """
    if dataset_str.endswith('.tsv'):
        dataset = datasets.load_dataset("csv", data_files=dataset_str, delimiter="\t", num_proc=32)['train']
    elif dataset_str.endswith('.jsonl'):
        dataset = datasets.load_dataset("json", data_files=dataset_str)
    else:
        dataset = datasets.load_from_disk(dataset_str)
    return dataset


def parse_dataset_with_truncation(dataset_str, dataset_name, max_length=1600):
    """Parse dataset and optionally truncate long documents.
    
    For non-MTR/TopiOCQA datasets, truncates 'text' field to max_length.
    
    Args:
        dataset_str: Path to dataset file or directory.
        dataset_name: Name of the dataset (used for truncation logic).
        max_length: Maximum text length for truncation. Default 1600.
        
    Returns:
        A HuggingFace Dataset object.
    """
    dataset = parse_dataset(dataset_str)
    
    # Some documents are extremely long; truncate for non-MTR/TopiOCQA datasets
    if 'mtr' not in dataset_name.lower() and 'topiocqa' not in dataset_name.lower():
        dataset = dataset.map(
            lambda x: {'text': x['text'][:max_length] if x.get('text') else ""},
            num_proc=32, load_from_cache_file=False, desc="length truncate"
        )
    return dataset
