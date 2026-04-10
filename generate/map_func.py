
import random
import re
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prompt import *
import seed_prompt
from loguru import logger
from shared.map_utils import merge_by_add, merge_by_replace, merge_by_append

# topic ctxs answers ground_truth_ctx messages
def make_query(instances, tokenizer):
    ctx = instances['ctxs']
    
    # Keep original logic, randomly select 5 documents
    ctx = random.sample(ctx, k=5)

    document_list = [item['text'] for item in ctx]
    cleaned_documents = [re.sub(r'\[\d+\]', '', doc) for doc in document_list]
    document_str = '\n---\n'.join([f"[{i+1}] {doc}" for i, doc in enumerate(cleaned_documents)])

    selected_did = [item['document_idx'] for item in ctx]

    # When there are no history messages, logic stays the same
    if 'messages' not in instances or not instances['messages']:
 # Assuming seed_prompt.QUERY_WO_HISTORY QUERY_WO_HISTORY 
        SEED = random.choice(seed_prompt.QUERY_WO_HISTORY)
        PROMPT = QUERY_WO_HISTORY
        
        prompt = PROMPT.format_map({'SEED': SEED, 'DOCUMENTS': document_str})

        token_ids = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': '**Question:**\n['}],
            add_generation_prompt=False,
            continue_final_message=True
        )
        
    # When there are history messages, use new prompt and logic
    else:
        SEED = random.choice(seed_prompt.CONVERSATION_W_HISTORY)
        PROMPT = QUERY_W_HISTORY

        # 1. Format history messages as a complete conversation
        history_str = ""
        previous_questions = []
        for message in instances['messages']:
            role_prefix = "User" if message['role'] == 'user' else "Assistant"
            history_str += f"{role_prefix}: {message['content']}\n"
            # Collect all previous user questions for anti-repetition
            if message['role'] == 'user':
                previous_questions.append(message['content'])

        # 2. Format previous questions as a numbered list
        if previous_questions:
            prev_q_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(previous_questions)])
        else:
            prev_q_str = "(None)"

        # 3. Fill in the new prompt template
        prompt = PROMPT.format_map({
            'SEED': SEED, 
            'DOCUMENTS': document_str, 
            'HISTORY': history_str.strip(),
            'PREVIOUS_QUESTIONS': prev_q_str
        })
        
        # 3. Build model input
        token_ids = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': '**Question:**\n['}],
            add_generation_prompt=False,
            continue_final_message=True
        )

    prompt_tokens = len(token_ids)
    
    return {'input_ids': token_ids, 'selected_did': selected_did, 'prompt_tokens': prompt_tokens}










def split_query(instance):
    """
    Extract query information from instance and build message list.
    
    Args:
        instance: Dictionary containing original question and context information
        
    Returns:
        dict: Dictionary containing the following keys:
            - ground_truth_document_idx: Ground truth document index
            - query: Extracted question
            - ground_truth_ctx: Ground truth context
            - messages: Message history
    """
    # Initialize default return values
    result = {
        'ground_truth_document_idx': -1,
        'query': "drop",
        'ground_truth_ctx': None,
        'messages': []
    }
    
    # Process raw text
    text = instance.get('raw_question', '')
    if '\n</think>\n\n' in text:
        text = text.split('\n</think>\n\n')[-1]
    
    # Try to extract question and ID
    # pattern = r'\[(\d+)\]\s*(.+)'
    pattern = r'(\d+)\]\s*(.+)'
    matches = re.findall(pattern, text)
    
    if not matches:
        logger.warning(f'No pattern match found in text: {text}')
        return result
    
    try:
        did, query = matches[0] # Turn match
        result['query'] = query.strip()
        
        # GetGround truth document index
        selected_dids = instance.get('selected_did', [])
        if selected_dids and did.isdigit():
            idx = int(did) - 1
            if 0 <= idx < len(selected_dids):
                result['ground_truth_document_idx'] = selected_dids[idx]
                
                # Find corresponding context
                ctx_list = instance['ctxs']
                result['ground_truth_ctx'] = next(
                    (ctx for ctx in ctx_list 
                     if ctx.get('document_idx') == result['ground_truth_document_idx']),
                    None
                )
                assert result['ground_truth_ctx'] is not None
    except (IndexError, ValueError, KeyError) as e:
        logger.warning(f'Error processing instance: {e}, DID: {did}, Query: {query}')
    
    # Build message list
    try:
        result['messages'] = instance['messages'] + [{'role': 'user', 'content': result['query']}]
    except KeyError:
        result['messages'] = [{'role': 'user', 'content': result['query']}]
    
    return result

# merge_by_add, merge_by_replace, merge_by_append imported from shared.map_utils


def make_response(instances,tokenizer):
    
    ground_truth_document = instances['ground_truth_ctx']['text']
    document_str = re.sub(r'\[\d+\]', '', ground_truth_document) 

    prompt = RESPONSE.format_map({'DOCUMENTS':document_str,'QUESTION':instances['messages'][-1]['content']})

    # if 'messages' not in instances:
    #     token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True)
    # else:
    token_ids = tokenizer.apply_chat_template(instances['messages'][:-1]+[{'role':'user','content':prompt}],add_generation_prompt=True)

    prompt_tokens = len(token_ids) + instances['prompt_tokens']

    return {'input_ids':token_ids,'prompt_tokens':prompt_tokens}

