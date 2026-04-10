import math
import numpy as np


def recall(ranked_indices_list, gold_index_list, topk):
    hit = 0
    
    for ranked_indices, gold_index in zip(ranked_indices_list, gold_index_list):
        
        for idx in ranked_indices[:topk]:
            try:
                if int(idx) == int(gold_index):
                    hit += 1
                    break

            except:
    recall = hit / len(ranked_indices_list)
    return recall
    
def mrr(ranked_indices_list, gold_index_list, topk):
    """
    Calculates Mean Reciprocal Rank (MRR)@k.
    Assumes one gold item per query.

    Args:
        ranked_indices_list (list of list of int/str): 
            A list where each inner list contains ranked item indices for a query.
        gold_index_list (list of int/str): 
            A list where each element is the gold (relevant) item index for the corresponding query.
        topk (int): The cut-off for evaluation. If the gold item is found beyond topk, 
                    its reciprocal rank for this query is considered 0.

    Returns:
        float: MRR@k score.
    """
    if not ranked_indices_list or not gold_index_list:
        return 0.0
    if len(ranked_indices_list) != len(gold_index_list):
        raise ValueError("ranked_indices_list and gold_index_list must have the same length.")
    if topk <= 0: # MRR@0 is typically 0
        return 0.0
        
    total_reciprocal_rank = 0.0
    num_queries = len(ranked_indices_list)

    for ranked_indices, gold_index in zip(ranked_indices_list, gold_index_list):
        try:
            str_gold_index = str(gold_index)
            for i, idx in enumerate(ranked_indices[:topk]): # Only consider up to topk
                if str(idx) == str_gold_index:
                    total_reciprocal_rank += 1.0 / (i + 1) # Rank is i+1 (1-indexed)
                    break # Found the first relevant item for this query
        except Exception as e:
            print(f"Error processing for MRR: ranked_indices={ranked_indices}, gold_index={gold_index}. Error: {e}")
            continue

    mrr_score = total_reciprocal_rank / num_queries if num_queries > 0 else 0.0
    return mrr_score


def ndcg(ranked_indices_list, gold_index_list, topk):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG)@k.
    Assumes one gold item per query (binary relevance: 1 if gold, 0 otherwise).

    Args:
        ranked_indices_list (list of list of int/str): 
            A list where each inner list contains ranked item indices for a query.
        gold_index_list (list of int/str): 
            A list where each element is the gold (relevant) item index for the corresponding query.
        topk (int): The cut-off for evaluation.

    Returns:
        float: NDCG@k score.
    """
    if not ranked_indices_list or not gold_index_list:
        return 0.0
    if len(ranked_indices_list) != len(gold_index_list):
        raise ValueError("ranked_indices_list and gold_index_list must have the same length.")
    if topk <= 0: # NDCG@0 is 0
        return 0.0

    total_ndcg = 0.0
    num_queries = len(ranked_indices_list)

    for ranked_indices, gold_index in zip(ranked_indices_list, gold_index_list):
        dcg_at_k = 0.0
        try:
            str_gold_index = str(gold_index)
            for i, idx in enumerate(ranked_indices[:topk]): # Iterate up to topk
                rank = i + 1 # Rank is 1-indexed
                if str(idx) == str_gold_index:
                    # Gold item found at this rank
                    # Relevance is 1 for the gold item
                    dcg_at_k += 1.0 / math.log2(rank + 1) # Discount factor for rank
                    break # Since there's only one gold item, its contribution is found
        except Exception as e:
            print(f"Error processing for NDCG (DCG part): ranked_indices={ranked_indices}, gold_index={gold_index}. Error: {e}")
            continue
            
        # Calculate IDCG@k
        # For a single relevant item, the ideal ranking places it at rank 1.
        # IDCG@k = relevance_of_gold_item / log2(1 + 1)
        # Here, relevance_of_gold_item is 1.
        # So, IDCG@k = 1.0 / log2(2) = 1.0, provided k >= 1.
        # If k=0, IDCG=0. If no relevant items (not our case here), IDCG=0.
        idcg_at_k = 1.0 / math.log2(1 + 1) if topk >= 1 else 0.0 
        # A more robust IDCG would be 0 if gold_index is not actually supposed to be in the list,
        # but here gold_index_list implies it is relevant for the query.

        if idcg_at_k > 0:
            total_ndcg += dcg_at_k / idcg_at_k
        # If idcg_at_k is 0 (e.g., topk=0), ndcg for this query is 0.
        
    ndcg_score = total_ndcg / num_queries if num_queries > 0 else 0.0
    return ndcg_score
