import json
import argparse
import datasets

def parse_args():

 parser = argparse.ArgumentParser()
 parser.add_argument('--dataset_dir')
 parser.add_argument('--document_dir')
 parser.add_argument('--output_dir')
 return parser.parse_args()

if __name__=='__main__':
 args = parse_args()
 data = json.load(open(args.dataset_dir))

 document = datasets.load_dataset('json',data_files=args.document_dir)['train']
 document = document.rename_column("ref_id", "document_idx").rename_column("ref_string", "ctx")
 all_data = []
 for item in data:
 convs = item['turns']
 previous_messages=[]
 for turn in convs:
 previous_messages +=[{'content':turn['question'],'role':'user'},{'role':'assistant','content':turn['response']}]
 # For simplicity, we only validate data with a unique gold document.
 if len(turn['golden_docs_pids'])==1:
 temp_data={}
 temp_data['messages'] = previous_messages.copy()

 temp_data['ground_truth_document_idx'] = turn['golden_docs_pids']
 golden_docs_pid=turn['golden_docs_pids'][0]
 target_idx = golden_docs_pid-1
 temp_data['ground_truth_ctx'] = document[target_idx]

 # Find 7 nearby documents to form 8 total for tag scoring.
 num_ctxs_to_fetch = 8
 
 start_idx = target_idx - (num_ctxs_to_fetch // 2 - 1)
 end_idx = start_idx + num_ctxs_to_fetch

 # Adjust boundary: if start index < 0
 if start_idx < 0:
 start_idx = 0
 end_idx = num_ctxs_to_fetch # Adjust end index to required count
 # Adjust boundary: if end index exceeds total ()
 elif end_idx > len(document):
 end_idx = len(document) # Set end index to total
 start_idx = len(document) - num_ctxs_to_fetch # Adjust start index back


 temp_data['ctxs'] = document.select(range(start_idx,end_idx)).to_list()
 all_data.append(temp_data) 

 dataset = datasets.Dataset.from_list(all_data)
 dataset.save_to_disk(args.output_dir)