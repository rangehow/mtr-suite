"""
    Judge prompts for analysis of previous benchmarks.
    Imports from shared/judge_prompts.py to avoid duplication.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.judge_prompts import (
    JUDGE_QUERYER_TAG_DOCUMENT_CORRECT_PROMPT,
    JUDGE_QUERY_RELATED_TO_CORRECT_DOCUMENT_PROMPT,
    JUDGE_QUERY_ATOMICITY,
    JUDGE_QUERY_Explicit_Reference,
    JUDGE_RESPONSE_FAITHFUL_PROMPT,
    JUDGE_ANSWER_QUALITY_PROMPT,
)
