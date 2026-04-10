""" 
Some weird operation in this script is to be consistant with setting in chatqa work,
see https://huggingface.co/nvidia/dragon-multiturn-query-encoder/blob/evaluation/evaluate.py
"""
import datasets
import json
import argparse
import os # For os.path.join

def main(args):
    # process query
    avaliable_dataset_query = ['doc2dial','inscit','qrecc','quac','topiocqa']
    for dataset_name in avaliable_dataset_query:
        output_query_dataset_path = os.path.join(args.output_datasets_path, dataset_name)
        if os.path.exists(output_query_dataset_path):
            continue
        print(f"Processing query for dataset: {dataset_name}")
        # Path to the dataset loading script or data files for ChatRAG-Bench
        if dataset_name == 'topiocqa':
 # needjson(Containsgold_idx dataloadentries)
            data = json.load(open(args.topiocqa_modify_dir))
            dataset = datasets.Dataset.from_list(data)
        else:
            dataset_load_path = args.chatrag_bench_path 
            
            dataset = datasets.load_dataset(dataset_load_path, dataset_name)

        
        print(f"Saving query dataset {dataset_name} to: {output_query_dataset_path}")
        dataset.save_to_disk(output_query_dataset_path)
        print(f"Successfully saved query dataset {dataset_name}")

    # process document
    avaliable_dataset_doc = ['topiocqa','qrecc','quac','doc2dial',]

    for dataset_name in avaliable_dataset_doc:
        print(f"\nProcessing documents for dataset: {dataset_name}")
        output_document_dataset_path = os.path.join(args.output_datasets_path, f'{dataset_name}_document')
        output_domain_map_path = os.path.join(output_document_dataset_path, 'domain_map.json')

        # if os.path.exists(output_document_dataset_path):
        #     continue

        if dataset_name == 'topiocqa':
            data = datasets.load_dataset("csv", data_files=args.topiocqa_doc_dir, delimiter="\t", num_proc=32)['train']
            data = data.map(lambda x:{'text':f"{x['title']}\n{x['text']}"},num_proc=64,remove_columns=data.column_names)
            data.save_to_disk(output_document_dataset_path) 

            
        else:
            document_json_path = os.path.join(args.chatrag_bench_path, "data", dataset_name, "documents.json")
            print(f"Loading documents from: {document_json_path}")
            with open(document_json_path, 'r') as f:
                data = json.load(f)
            
            if dataset_name == 'qrecc':
                query_dataset_load_path = os.path.join(args.output_datasets_path, dataset_name)
                print(f"Loading processed query dataset from: {query_dataset_load_path}")
                query_dataset = datasets.load_from_disk(query_dataset_load_path)
            
            if 'test' in query_dataset:
                query_dataset = query_dataset['test']
            else:
                query_dataset = query_dataset['dev']
            
    
            if dataset_name=='qrecc':
                print("Modifying qrecc data based on ground truth...")
                for item in query_dataset:
                    gold_idx = item['ground_truth_ctx']['index']
                    doc_id = item['document']
                    answer = item['answers'][0]
                    # Ensure doc_id exists and gold_idx is within bounds
                    if doc_id in data and isinstance(data[doc_id], list) and 0 <= gold_idx < len(data[doc_id]):
                        data[doc_id][gold_idx] =  answer + " || " +  data[doc_id][gold_idx]
                    else:
                        print(f"Warning: Could not find doc_id '{doc_id}' or gold_idx '{gold_idx}' for qrecc modification.")


            all_data ={'text':[]}
            domain_map = {}
            current_idx = 0

            print("Processing domain map...")
            for domain,texts in data.items():
                domain_map[domain] = (current_idx,current_idx+len(texts))
                current_idx += len(texts)
                all_data['text'].extend(texts)

            # These prints are for debugging, can be kept or removed
            # print("Comparing query dataset documents with domain map keys:")
            # print(f"  Query dataset documents == domain_map keys: {set(query_dataset['document']) == set(domain_map.keys())}")
            # print(f"  In query_dataset but not in domain_map: {set(query_dataset['document']) -set(domain_map.keys())}")
            # print(f"  In domain_map but not in query_dataset: {set(domain_map.keys()) - set(query_dataset['document'])}")
            
            dataset_docs = datasets.Dataset.from_dict(all_data)

            

            print(f"Saving document dataset to: {output_document_dataset_path}")
            dataset_docs.save_to_disk(output_document_dataset_path) # Uncomment if needed
            print(f"Saving domain map to: {output_domain_map_path}")
            os.makedirs(output_document_dataset_path, exist_ok=True) # Ensure directory exists for domain_map.json
            with open(output_domain_map_path,'w') as f_map:
                json.dump(domain_map, f_map, ensure_ascii=False,indent=2) # Uncomment if needed
            print(f"Successfully processed documents for {dataset_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process datasets and documents, separating path information.")
    parser.add_argument('--chatrag_bench_path', type=str, required=True,
                        help='Base path to the ChatRAG-Bench dataset directory.')
    parser.add_argument('--output_datasets_path', type=str, required=True,
                        help='Base path for saving processed datasets.')
    parser.add_argument('--topiocqa_modify_dir', type=str, required=True,
                        help='path for loading json from add_gold_idx_to_topiocqa')
    parser.add_argument('--topiocqa_doc_dir', type=str, required=True,)
    parsed_args = parser.parse_args()
    main(parsed_args)