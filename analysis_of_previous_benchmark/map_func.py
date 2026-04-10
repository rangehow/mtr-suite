
import random
import re
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prompt import *
from loguru import logger
from shared.map_utils import merge_by_add, merge_by_replace, merge_by_append

def mock_chatrag(instance):

    messages = instance['messages']
    
    minimize_length=10000000
    answers = instance['answers']
    if isinstance(instance['answers'],list):
        for answer in instance['answers']:
            if len(answer)<minimize_length:
                minimize_length = len(answer)
                answers = answer

    messages = messages + [{'content': answers, 'role': 'assistant'}]
    return {'messages':messages}


def make_judge(instances,tokenizer):


    ground_truth_document = instances['ground_truth_ctx'][0]['ctx'] if isinstance(instances['ground_truth_ctx'],list) else instances['ground_truth_ctx']['ctx']

    document_str = re.sub(r'\[\d+\]', '', ground_truth_document) 

    prompt = JUDGE_QUERYER_TAG_DOCUMENT_CORRECT_PROMPT.format_map({'DOCUMENT':document_str,'QUESTION':instances['messages'][0]['content']})

    token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':'**Justification:**'}],add_generation_prompt=False,continue_final_message=True)

    return {'input_ids':token_ids}




def make_find_document(instances, tokenizer):
    """
    Selects up to 8 documents from instances['ctxs'], ensuring the ground truth
    document is included if present, and prepares the prompt for the model.

    Args:
        instances (dict): A dictionary containing 'ctxs' and potentially
                          'ground_truth_ctx', and 'messages'.
        tokenizer: The tokenizer to use for applying chat template.

    Returns:
        dict: A dictionary containing the input_ids for the model.
    """

    all_ctxs = instances['ctxs']
    selected_ctxs = []
    ground_truth_document_text = None
    ground_truth_item_in_ctxs = None

    # 1. Get the text of the ground truth document if it exists
    if 'ground_truth_ctx' in instances and instances['ground_truth_ctx']:
        gt_ctx_data = instances['ground_truth_ctx']
        if isinstance(gt_ctx_data, list) and gt_ctx_data:
            # Assuming the first item in the list is the relevant one
            ground_truth_document_text = gt_ctx_data[0].get('ctx')
        elif isinstance(gt_ctx_data, dict):
            if 'ctx' in gt_ctx_data:
                ground_truth_document_text = gt_ctx_data.get('ctx')
            elif 'text' in gt_ctx_data:
                ground_truth_document_text = gt_ctx_data.get('text')

        
    if ground_truth_document_text:
        for item in all_ctxs:
            if item.get('ctx') == ground_truth_document_text: # Assuming 'text' key in ctxs items holds the text
                ground_truth_item_in_ctxs = item
                break

    # 3. Add ground truth document first if found in all_ctxs
    if ground_truth_item_in_ctxs:
        selected_ctxs.append(ground_truth_item_in_ctxs)

    # 4. Add other documents, prioritizing those appearing earlier in all_ctxs, up to 8 total
    for item in all_ctxs:
        # Check if we have space (less than 8 documents selected)
        # and if this item is NOT the ground truth item (to avoid duplicates if GT was added)
        if len(selected_ctxs) < 8:
            # Only add if ground_truth_item_in_ctxs was not found, OR
            # if it was found, only add if this item is not the same object reference
            if ground_truth_item_in_ctxs is None or item is not ground_truth_item_in_ctxs:
                 selected_ctxs.append(item)
        else:
            break # Stop adding once we reach 8 documents

    # If ground truth was not in all_ctxs, this loop correctly takes the first 8.
    # If ground truth was in all_ctxs, it adds the GT first, then takes up to 7 others from the start, skipping the GT item.

    # 5. Proceed with the rest of the original logic using the selected_ctxs
    document_list = [item['ctx'] for item in selected_ctxs] # Extract text from selected items
    cleaned_documents = [re.sub(r'\[\d+\]', '', doc) for doc in document_list]
    # Ensure documents are not empty after cleaning, though unlikely for ctxs
    cleaned_documents = [doc for doc in cleaned_documents if doc.strip()]

    # Handle case where selected_ctxs might be empty (e.g., all_ctxs was empty)
    if not cleaned_documents:
        document_str = ""
    else:
        document_str = '\n---\n'.join([f"[{i+1}] {doc}" for i, doc in enumerate(cleaned_documents)])


    # Make sure 'messages' key exists and has content
    question_text = ""
    if 'messages' in instances and isinstance(instances['messages'], list) and instances['messages']:
         question_text = instances['messages'][0].get('content', '')


    # JUDGE_QUERY_RELATED_TO_CORRECT_DOCUMENT_PROMPT must be defined
    # Check if the prompt template constant is defined
    if 'JUDGE_QUERY_RELATED_TO_CORRECT_DOCUMENT_PROMPT' not in globals():
         raise NameError("JUDGE_QUERY_RELATED_TO_CORRECT_DOCUMENT_PROMPT is not defined.")

    prompt = JUDGE_QUERY_RELATED_TO_CORRECT_DOCUMENT_PROMPT.format_map({'DOCUMENTS': document_str, 'QUESTION': question_text})

    token_ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': '**Justification:**'}],
        add_generation_prompt=False,
        continue_final_message=True
    )


    return {'input_ids': token_ids,'selected_ctxs':selected_ctxs,'ground_truth_item_in_ctxs':ground_truth_item_in_ctxs}


def make_atomic(instances,tokenizer):

    
    prompt = JUDGE_QUERY_ATOMICITY.format_map({'QUESTION':instances['messages'][0]['content']})
    token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':'**Justification:**'}],add_generation_prompt=False,continue_final_message=True)

    return {'input_ids':token_ids}


def make_explicit_reference(instances,tokenizer):

    
    prompt = JUDGE_QUERY_Explicit_Reference.format_map({'QUESTION':instances['messages'][0]['content']})
    token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':'**Justification:**'}],add_generation_prompt=False,continue_final_message=True)

    return {'input_ids':token_ids}



def make_response_faithful(instances,tokenizer):

    
    prompt = JUDGE_RESPONSE_FAITHFUL_PROMPT.format_map({'QUESTION':instances['messages'][0]['content'],'RESPONSE':instances['messages'][1]['content'],'DOCUMENT':instances['ground_truth_ctx'][0]['ctx'] if isinstance(instances['ground_truth_ctx'],list) else instances['ground_truth_ctx']['ctx'] })
    token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':'**Justification:**'}],add_generation_prompt=False,continue_final_message=True)

    return {'input_ids':token_ids}


def make_response_quality(instances,tokenizer):

    
    prompt = JUDGE_ANSWER_QUALITY_PROMPT.format_map({'QUESTION':instances['messages'][0]['content'],'RESPONSE':instances['messages'][1]['content'],'DOCUMENT':instances['ground_truth_ctx'][0]['ctx'] if isinstance(instances['ground_truth_ctx'],list) else instances['ground_truth_ctx']['ctx'] })
    token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':'**Justification:**'}],add_generation_prompt=False,continue_final_message=True)

    return {'input_ids':token_ids}

