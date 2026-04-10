import json
from utils import *
from arg_parser import parse_args
from transformers import AutoTokenizer
from datasets import load_from_disk,load_dataset
from sharegpt import load_sharegpt_dataset,analyze_sharegpt_conversation_stats,analyze_sharegpt_response_type
from domain_classify import classify_document_domain
import multiprocessing
if __name__=='__main__':
    args = parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)


    # coral_document = load_dataset('json',data_files=args.coral_document_dir)['train']
    # get_avg_tokens_of_dataset(coral_document,tokenizer,'ref_string')
    # 


    # coral_dataset = load_dataset('json',data_files=args.coral_dataset_dir)['train']
    # get_coral_response_length(coral_dataset,tokenizer)
    # get_turns_of_dataset(coral_dataset)
    


    mtr_test = load_from_disk(args.mtr_test_path)
    get_mtr_token_length(mtr_test,tokenizer)
    # get_mtr_topic(mtr_test)
    mtr_train = load_from_disk(args.mtr_train_path)
    get_mtr_token_length(mtr_train,tokenizer)
    # get_mtr_topic(mtr_train)
    mtr_total = datasets.concatenate_datasets([mtr_train,mtr_test])
    get_mtr_token_length(mtr_total,tokenizer)
    # get_mtr_topic(mtr_total)
    print(len(mtr_test),len(mtr_train),len(mtr_total))
    # processed_dataset = load_from_disk(args.processed_dataset_path)
    # get_avg_tokens_of_dataset(processed_dataset,tokenizer)

    # sharegpt
    # sharegpt_dataset = load_dataset(args.sharegpt_path)['train']
    # analyze_sharegpt_conversation_stats(sharegpt_dataset,tokenizer)
    # analyze_sharegpt_response_type(sharegpt_dataset)

    # mtr_domain
    # mtr_document = load_from_disk(args.processed_dataset_path)
    # classify_document_domain(mtr_document,args.domain_classifier_path,args.domain_result_path)