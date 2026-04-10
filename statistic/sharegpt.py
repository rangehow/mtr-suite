
from functools import partial
import datasets
import numpy as np
from tqdm import tqdm

def load_sharegpt_dataset(sharegpt_path):

 import json
 import os
 data_part1 = json.load(open(os.path.join(sharegpt_path,'sg_90k_part1.json')))

 from tqdm import tqdm

 #33020
 # 38285
 datasets.Dataset.from_list([data_part1[33020]]+[data_part1[38285]])
 for end in tqdm(range(38500,38000,-1)):
 try:
 dataset = datasets.Dataset.from_list(data_part1[33020:end])
 
 except Exception as e:
...

 data_part2 = json.load(open(os.path.join(sharegpt_path,'sg_90k_part2.json')))
 return dataset


def analyze_sharegpt_conversation_stats(sharegpt_dataset,tokenizer):
 """
 ShareGPT,.

 Args:
 conversations_data:, ().
: [{'messages': [{'from': 'human', 'value': '...'}, {'from': 'gpt', 'value': '...'},...]},...]

 Returns:
,:
 'average_turns_per_conversation': float,
 'average_response_length': float, # GPT 
 'total_conversations': int, # 
 'total_responses': int # GPT 
 """
 # 
 

 def tokenize(instance,tokenizer):

 assistant_token_length=0
 cnt=0
 user_token_length = 0
 conversations = instance['conversations']
 for role,value in zip(conversations['from'],conversations['value']):
 if role =='system':
 continue
 
 elif role == 'human' or role=='user':
 user_token_length+=len(tokenizer.encode(value))
 elif role =='gpt' or role=='chatgpt' or role=='bing' or role == 'bard' or role=='assistant':
 assistant_token_length+=len(tokenizer.encode(value))
 else:
 print(role)
 return {'user_token':user_token_length,'assistant_token':assistant_token_length,'turn_length':len(conversations['from'])//2}
 

 dataset = sharegpt_dataset.map(partial(tokenize,tokenizer=tokenizer),num_proc=32)

 turn_cnt = 0
 token_cnt = 0
 valid_data_cnt = 0
 blank_cnt = 0
 for data in dataset:
 if data['turn_length'] == 0:
 # print(data['user_token'])
 blank_cnt+=1
 continue
 if data['user_token']/data['turn_length']<20:
 valid_data_cnt +=1
 turn_cnt+=data['turn_length']
 token_cnt +=data['assistant_token']

 # turn_cnt = np.mean(dataset['turn_length'])
 # token_length=np.mean(dataset['assistant_token'])
 print(blank_cnt)
 print(turn_cnt/valid_data_cnt)
 print(token_cnt/turn_cnt)
# 7.97968345006342
# 463.69771713859996
def analyze_sharegpt_response_type(sharegpt_dataset):
...

if __name__=='__main__':
...