"""
Embedding utilities - delegates to shared.data_utils for parse_dataset.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.data_utils import parse_dataset

__all__ = ['parse_dataset']
