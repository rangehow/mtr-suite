"""
Shared map utility functions used by generate/, model_choice/, and analysis_of_previous_benchmark/.
"""


def merge_by_add(instance, index, key, data):
    """Merge by adding (concatenating) data to existing value."""
    return {key: instance[key] + data[index]}


def merge_by_replace(instance, index, key, data):
    """Merge by replacing existing value with new data."""
    return {key: data[index]}


def merge_by_append(instance, index, key, data):
    """Merge by appending to existing list. Only supports list values."""
    return {key: instance[key] + [data[index]]}
