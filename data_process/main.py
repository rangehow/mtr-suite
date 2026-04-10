from datasets import load_dataset,load_from_disk
from arg_parser import parse_args
from filter import _remove_useless_part_in_wiki,filter_by_classifier,generate_tag_by_classifier,generate_tag_by_nvidia_finewebedu_classifier
from chunking import group_texts
from minhash import dedup
from functools import partial


def sequential_map(dataset,funcs):
    for func in funcs:
        print(func)
        if func['type'] == 'map':
            if 'remove' in func and func['remove']:
                dataset = dataset.map(
                    func['func'],
                    num_proc=64,
                    batched=True,
                    load_from_cache_file=False,
                    remove_columns=dataset.column_names,
                    # desc = func['func'].__name__
                )
            else:
                dataset = dataset.map(
                    func['func'],
                    num_proc=64,
                    batched=True,
                    load_from_cache_file=False,
                )
                
        else:
            dataset = func['func'](dataset)
    
    return dataset

def main():
    # parseparameter
    args = parse_args()
    
    chunk_func = partial(group_texts, max_length = args.max_length)
    filter_by_classifier_func = partial(filter_by_classifier, model_path = args.quality_model_path,worker_func=generate_tag_by_classifier)
    filter_by_finewebedu_func = partial(filter_by_classifier, model_path = args.fineweb_model_path,worker_func=generate_tag_by_nvidia_finewebedu_classifier)
    funcs=[
        {'type':'map','func':_remove_useless_part_in_wiki}, # A heuristic text extraction function specifically designed for Wikipedia's structure.
        {'type':'map','func':chunk_func,'remove':True}, # A function for efficiently chunking text by a preset length ( seperator based ).
        {'type':'non-map','func':dedup}, # Large-scale text deduplication function based on MinHash-LSH.
        {'type':'non-map','func':filter_by_finewebedu_func}, # A function for labeling data using a scalar classifier and filtering the data according to a predefined threshold.
        {'type':'non-map','func':filter_by_classifier_func}, # A function for labeling data using a scalar classifier and filtering the data according to a predefined threshold.
        
    ]
    # usingparameter
    try:
        dataset = load_dataset(args.dataset_path)['train']
    except:
        try:
            dataset = load_from_disk(args.dataset_path)
        except Exception as e:
            print(f"load dataset failed {e}")
    print(dataset)
    print(dataset[0])
    # start to process dataset, specific function can be viewed in func dict.
    dataset = sequential_map(dataset,funcs)
    print(dataset)
    print("the dataset len is:")
    try:
        print(len(dataset.unique('id')),len(dataset))
    except:
        print(len(dataset))
    dataset.save_to_disk(f'{args.save_dir}',num_proc=64)
    
import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()