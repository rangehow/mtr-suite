
import random
import re
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prompt import *
from loguru import logger
from shared.map_utils import merge_by_add, merge_by_replace, merge_by_append


def make_judge(instances,tokenizer):
    
    ground_truth_document = instances['ground_truth_ctx']['text']
    document_str = re.sub(r'\[\d+\]', '', ground_truth_document) 

    prompt = JUDGE_QUERYER_TAG_DOCUMENT_CORRECT_PROMPT.format_map({'DOCUMENT':document_str,'QUESTION':instances['messages'][0]['content']})

    token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':'**Justification:**'}],add_generation_prompt=False,continue_final_message=True)

    return {'input_ids':token_ids}


def make_find_document(instances,tokenizer):

    ctx = instances['ctxs']
    document_list = [item['text'] for item in ctx]
    cleaned_documents = [re.sub(r'\[\d+\]', '', doc) for doc in document_list]
    document_str = '\n---\n'.join([f"[{i+1}] {doc}" for i, doc in enumerate(cleaned_documents)])

    prompt = JUDGE_QUERY_RELATED_TO_CORRECT_DOCUMENT_PROMPT.format_map({'DOCUMENTS':document_str,'QUESTION':instances['messages'][0]['content']})

    token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':'**Justification:**'}],add_generation_prompt=False,continue_final_message=True)

    return {'input_ids':token_ids}


def make_atomic(instances,tokenizer):

    
    prompt = JUDGE_QUERY_ATOMICITY.format_map({'QUESTION':instances['messages'][0]['content']})
    token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':'**Justification:**'}],add_generation_prompt=False,continue_final_message=True)

    return {'input_ids':token_ids}


def make_explicit_reference(instances,tokenizer):

    
    prompt = JUDGE_QUERY_Explicit_Reference.format_map({'QUESTION':instances['messages'][0]['content']})
    token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':'**Justification:**'}],add_generation_prompt=False,continue_final_message=True)

    return {'input_ids':token_ids}



def make_response_faithful(instances,tokenizer):

    
    prompt = JUDGE_RESPONSE_FAITHFUL_PROMPT.format_map({'QUESTION':instances['messages'][0]['content'],'RESPONSE':instances['messages'][1]['content'],'DOCUMENT':instances['ground_truth_ctx']['text']})
    token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':'**Justification:**'}],add_generation_prompt=False,continue_final_message=True)

    return {'input_ids':token_ids}


def make_response_quality(instances,tokenizer):

    
    prompt = JUDGE_ANSWER_QUALITY_PROMPT.format_map({'QUESTION':instances['messages'][0]['content'],'RESPONSE':instances['messages'][1]['content'],'DOCUMENT':instances['ground_truth_ctx']['text']})
    token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':'**Justification:**'}],add_generation_prompt=False,continue_final_message=True)

    return {'input_ids':token_ids}

