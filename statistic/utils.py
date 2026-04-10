from transformers import AutoTokenizer
import datasets
from tqdm import tqdm


def get_avg_tokens_of_dataset(dataset,tokenizer,text_key='text',sample_size=100000):

    dataset = dataset.shuffle()
    cnt=0
    char_length = 0
    token_lenght = 0
 # iterateentries
    for i in tqdm(range(sample_size)):
        cnt+=1
        char_length+=len(dataset[i][text_key])
        token_lenght+=len(tokenizer(dataset[i][text_key])['input_ids'])
        
    print(cnt)
    print(char_length)
    print(char_length/cnt)
    print(token_lenght)
    print(token_lenght/cnt)

# 100000
# 167870834
# 1678.70834
# 38953819
# 389.53819



def get_turns_of_dataset(dataset):
    dialogue_cnt = len(dataset)
    turn_cnt =0
    for i in range(len(dataset)):
        turn_cnt += len(dataset[i]['turns'])
    print(turn_cnt/dialogue_cnt)

def get_mtr_token_length(dataset,tokenizer):
    dialogue_cnt = len(dataset)//8
    response_length = 0
    query_length = 0
    for i in range(7,len(dataset),8):
        for j in range(1,len(dataset[i]['messages']),2):
            cur = len(tokenizer(dataset[i]['messages'][j]['content']).input_ids)
            response_length += cur
            # print(cur)
        for j in range(0,len(dataset[i]['messages']),2):
            cur = len(tokenizer(dataset[i]['messages'][j]['content']).input_ids)
            query_length += cur

    print('avg response length',response_length/dialogue_cnt/8)
    print('avg query length',query_length/dialogue_cnt/8)


def get_mtr_topic(dataset):
    topic_cnt = 0
    dialogue_cnt = len(dataset)//8
    for i in range(7,len(dataset),8):
        seen = set()
        for j in range(i-8,i):
            topic_idx = dataset[j]['ground_truth_ctx']['document_idx']
            seen.add(topic_idx)
        topic_cnt +=len(seen)
    
    print(topic_cnt/dialogue_cnt)
def get_coral_response_length(dataset,tokenizer):
    turn_cnt = 0
    response_length = 0
    for i in range(len(dataset)):
        turn_cnt +=len(dataset[i]['turns'])
        for turn in dataset[i]['turns']:
            cur = len(tokenizer(turn['response']).input_ids)
            response_length +=cur
            print(cur)
    print(response_length/turn_cnt)
