# This file's structure is not modular, but it works for now
from functools import partial
from transformers import AutoTokenizer
import numpy as np
import json
import datasets
import re
import os
from torch.cuda import device_count

from tqdm import tqdm
from map_func import *
from loguru import logger
import argparse
import sglang as sgl





def generate(llm,args,dataset,sampling_params):
    if args.inference_backend == 'sglang':
        outputs = llm.generate(input_ids=dataset['input_ids'], sampling_params=sampling_params)
    elif args.inference_backend == 'vllm':
        # According to function definition, prompt_token_ids will be deprecated in future versions
        outputs = llm.generate(prompt_token_ids=dataset['input_ids'], sampling_params=sampling_params)

    logger.info("generation process done")
    return outputs



def extract_from_outputs(args,outputs,tokenizer):
    if args.inference_backend == 'sglang':
        output_text=[item['text'] for item in outputs]
        completion_tokens=[item['meta_info']['completion_tokens'] for item in outputs]
    elif args.inference_backend == 'vllm':
        # According to function definition, prompt_token_ids will be deprecated in future versions
        output_text=[item.outputs[0].text for item in outputs]
        completion_tokens= tokenizer(output_text,return_length=True).length

    return output_text,completion_tokens



def extract_numeric_score_from_range_1_5(output_texts):
    # --- Optimization Start ---

    # Pre-compile regex patterns for speed
    # Pattern 1: Explicit "Rating: X" (case-insensitive)
    # Matches "**Rating**: 1", "Rating: 1", "Rating 1", "**Rating** 1" etc.
    pattern_explicit = re.compile(r"(?i)(?:\*\*Rating\*\*|Rating)\s*:?\s*([1-5])")

    # Pattern 2: Find standalone digits [1-5] as whole words (using word boundaries \b)
    # This helps avoid matching '1' in '10' or 'Item 1'. We'll find all and take the last.
    pattern_standalone_digit = re.compile(r"\b([1-5])\b")

    extracted_scores = [] # List to store the score for each output

    # Process each output text individually
    for text in output_texts:
        score = None  # Default score if no rating is found or conversion fails
        score_str = None

        # 1. Try to find the explicit "Rating: X" pattern anywhere in the text
        #    We search from the end using reversed finditer for efficiency if multiple exist,
        #    as the original logic took the last findall match.
        #    A simpler approach might be to just use search() if the *first* explicit rating is ok.
        #    Let's stick closer to the original logic: find *all* explicit matches and consider the last one.
        explicit_matches = pattern_explicit.findall(text)
        if explicit_matches:
            score_str = explicit_matches[-1] # Get the digit from the last explicit match
        else:
            # 2. If no explicit pattern, find all standalone digits [1-5]
            standalone_matches = pattern_standalone_digit.findall(text)
            if standalone_matches:
                # Take the *last* standalone digit found in the text
                score_str = standalone_matches[-1]

        # 3. Convert the found string digit to an integer
        if score_str:
            try:
                extracted_val = int(score_str)
                # Optional: Validate if the extracted value is indeed between 1 and 5
                if 1 <= extracted_val <= 5:
                    score = extracted_val
                else:
                     # This case should ideally not happen due to regex [1-5] but good for safety
                     logger.warning(f"Regex matched '{score_str}', but it's not in the range [1-5]. Text: {text}")
                     # score remains 0
            except ValueError:
                # This handles potential unexpected non-integer matches if regex were less strict
                logger.error(f"Could not convert extracted score '{score_str}' to integer. Text: {text}")
                # score remains 0
        else:
            # No rating found by either pattern
            logger.warning(f"Could not find rating score in output: {text}")
            # score remains 0

        extracted_scores.append(score)

    return extracted_scores
    # --- Optimization End ---


def extract_numeric_score_from_tag(output_texts,processed_dataset):
    
    
    regex_pattern = r'\[(\d+)\]$'
    compiled_regex = re.compile(regex_pattern)

    extracted_ids = []
    for i, text in enumerate(output_texts):
        extracted_number = None # Default to None
        try:
            if not isinstance(text, str):
                 logger.warning(f"Output item at index {i} is not a string (type: {type(text)}): '{str(text)[:100]}...' - Skipping ID extraction.")
                 # Keep extracted_number as None
            else:
                match = compiled_regex.search(text)
                if match:
                    extracted_number = match.group(1)
                # else: keep extracted_number as None (no match found)

        except Exception as e:
            # Catch unexpected errors during regex search or handling
            logger.error(f"Error processing output text at index {i}: '{str(text)[:100]}...' - Error: {e}", exc_info=True)
            # Keep extracted_number as None
        finally:
            # Append None even if an error occurred during processing this text
            extracted_ids.append(extracted_number)
    logger.info(f"Extracted {len(extracted_ids)} potential IDs (including None).")

    # 5. Calculate Scores based on extracted IDs
    extracted_scores = []
    logger.info("Calculating scores based on extracted IDs...")
    
    for i, extracted_id in enumerate(extracted_ids):
        score = None # Default score is None (e.g., if ID is None or errors occur)
        try:
            if extracted_id is not None:
                # Convert extracted ID string to integer index
                doc_index = int(extracted_id) - 1 # Potential ValueError

                # Access data safely
                record = processed_dataset[i] # Potential IndexError (less likely if lengths match check passed)
                selected_ctxs = record['selected_ctxs'] # Use .get() for safer dict access
                ground_truth_item_in_ctxs = record['ground_truth_item_in_ctxs']

                selected_ctx = selected_ctxs[doc_index]
                score = (selected_ctx == ground_truth_item_in_ctxs) # Calculate boolean score
                score = 1 if score else 0
        except Exception as e:
            # Catch any other unexpected error during score calculation for this record
            logger.error(f"Record {i}:error calculating score for extracted ID '{extracted_id}': {e}")
            # Score remains None
        
        extracted_scores.append(score)

    return extracted_scores

def extract_numeric_score_from_range_0_1(output_texts):


    regex_pattern = r"([01])\s*$"
    compiled_regex = re.compile(regex_pattern) # No need for MULTILINE or IGNORECASE here

    extracted_scores = []

    # Process each LLM output text
    for i, text in enumerate(output_texts):
        score = None # Default score is 0 (if no 0/1 found at the end or extraction fails)
        if text: # Check if the text is not empty
             # Search for the pattern at the end of the string
             match = compiled_regex.search(text)

             if match:
                 # Extract the captured digit ('0' or '1')
                 digit = match.group(1)
                 # Convert the digit string to an integer score
                 score = int(digit)
                 # logger.debug(f"Record {i}: Found final digit '{digit}', assigned score {score}") # Optional debug log
             else:
                 # Log if the expected pattern wasn't found at the end
                 # Log a snippet of the end of the text for easier debugging
                 log_snippet = text[-50:] # Show last 50 characters
                 logger.warning(
                     f"Record {i}: Could not extract final '0' or '1' from output. "
                     f"Assigning score 0. Output ends with: '...{log_snippet}'"
                 )
                 # score remains 0
        else:
             logger.warning(f"Record {i}: Received empty output text. Assigning score 0.")
             score = 0 # Ensure score is 0 for empty strings

        extracted_scores.append(score)
    
    return extracted_scores

def postprocess(extracted_scores,score_name,record_dataset,output_texts):
    # Add the extracted scores as a new column
    # Ensure the number of scores matches the number of records in record_dataset
    if  len(extracted_scores) == len(record_dataset):
        if f'{score_name}_reason' in record_dataset.column_names: # stupid code
            record_dataset = record_dataset.remove_columns(f'{score_name}_reason')
        if score_name in record_dataset.column_names:
            record_dataset = record_dataset.remove_columns(score_name)
        record_dataset = record_dataset.add_column(name=f'{score_name}_reason', column=output_texts)
        record_dataset = record_dataset.add_column(name=score_name, column=extracted_scores)
    else:
        logger.error(f"Mismatch between number of extracted scores ({len(extracted_scores)}) and dataset size ({len(record_dataset)}). Scores not added.")
        # Handle this error case appropriately, e.g., return original dataset or raise error

    return record_dataset


def preprocess(llm,args,sampling_params,tokenizer,dataset,map_func,score_name=None,record_dataset=None):
    # Prepare dataset

    # debug
    dataset = dataset.map(partial(map_func, tokenizer=tokenizer), num_proc=1, load_from_cache_file=False)
    # Generate outputs from LLM
    if f'{score_name}_reason' not in record_dataset.column_names:
        outputs = generate(llm, args, dataset, sampling_params)

        # Extract text outputs
        output_texts, completion_tokens = extract_from_outputs(args, outputs, tokenizer)
    else:
        output_texts =record_dataset[f'{score_name}_reason']
    return output_texts,dataset

import re
from functools import partial


def judge_answerable(dataset, llm, sampling_params, tokenizer, args, record_dataset, score_name):

    output_texts,dataset = preprocess(llm,args,sampling_params,tokenizer,dataset,make_judge,score_name,record_dataset)
    extracted_scores = extract_numeric_score_from_range_1_5(output_texts)
    record_dataset = postprocess(extracted_scores=extracted_scores,score_name=score_name,record_dataset=record_dataset,output_texts=output_texts)

    return record_dataset
    



def find_document(dataset, llm, sampling_params, tokenizer, args, record_dataset, score_name):

    
    output_texts,dataset = preprocess(llm,args,sampling_params,tokenizer,dataset,make_find_document,score_name,record_dataset)
    extracted_scores = extract_numeric_score_from_tag(output_texts,dataset)
    record_dataset = postprocess(extracted_scores=extracted_scores,score_name=score_name,record_dataset=record_dataset,output_texts=output_texts)

    return record_dataset





def judge_explicit_reference(dataset, llm, sampling_params, tokenizer, args, record_dataset, score_name):

    
    output_texts,dataset = preprocess(llm,args,sampling_params,tokenizer,dataset,make_explicit_reference,score_name,record_dataset)
    
    extracted_scores = extract_numeric_score_from_range_0_1(output_texts)

    record_dataset = postprocess(extracted_scores=extracted_scores,score_name=score_name,record_dataset=record_dataset,output_texts=output_texts)

    return record_dataset







def judge_atomic(dataset, llm, sampling_params, tokenizer, args, record_dataset, score_name):


    output_texts,dataset = preprocess(llm,args,sampling_params,tokenizer,dataset,make_atomic,score_name,record_dataset)
    
    extracted_scores = extract_numeric_score_from_range_0_1(output_texts)

    record_dataset = postprocess(extracted_scores=extracted_scores,score_name=score_name,record_dataset=record_dataset,output_texts=output_texts)

    return record_dataset







def judge_response_faithful(dataset, llm, sampling_params, tokenizer, args, record_dataset, score_name):

    output_texts,dataset = preprocess(llm,args,sampling_params,tokenizer,dataset,make_response_faithful,score_name,record_dataset)
    extracted_scores = extract_numeric_score_from_range_1_5(output_texts)
    record_dataset = postprocess(extracted_scores=extracted_scores,score_name=score_name,record_dataset=record_dataset,output_texts=output_texts)

    

    return record_dataset




def judge_response_quality(dataset, llm, sampling_params, tokenizer, args, record_dataset, score_name):

    output_texts,dataset = preprocess(llm,args,sampling_params,tokenizer,dataset,make_response_quality,score_name,record_dataset)
    extracted_scores = extract_numeric_score_from_range_1_5(output_texts)
    record_dataset = postprocess(extracted_scores=extracted_scores,score_name=score_name,record_dataset=record_dataset,output_texts=output_texts)

    return record_dataset