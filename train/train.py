import os
from typing import Any, Optional, Union
import torch
import datasets
from transformers import Trainer, AutoModel, AutoTokenizer, TrainingArguments
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
# torch.autograd.set_detect_anomaly(True)
import argparse
from datasets import disable_caching
from loss import infonce_loss
disable_caching()


def extend_position_embeddings(model, new_max_length):
 """
 BERT, position_ids token_type_ids buffer.

 Args:
 model: BERT (transformers.BertModel)
 new_max_length: 
 """
 if new_max_length <= model.config.max_position_embeddings:
 print(f" {new_max_length} {model.config.max_position_embeddings},.")
 return

 old_position_embeddings = model.embeddings.position_embeddings.weight.data
 old_max_length = model.config.max_position_embeddings
 embedding_dim = old_position_embeddings.size(1)

 # 
 new_position_embeddings = nn.Embedding(new_max_length, embedding_dim).weight.data
 torch.nn.init.normal_(new_position_embeddings, mean=0.0, std=model.config.initializer_range)

 # 
 n = min(old_max_length, new_max_length)
 new_position_embeddings[:n,:] = old_position_embeddings[:n,:]

 # 
 model.embeddings.position_embeddings = nn.Embedding.from_pretrained(new_position_embeddings)

 # **: position_ids token_type_ids buffer**
 model.embeddings.register_buffer(
 "position_ids", torch.arange(new_max_length).expand((1, -1)), persistent=False
)
 model.embeddings.register_buffer(
 "token_type_ids", torch.zeros((1, new_max_length), dtype=torch.long), persistent=False
)


 # 
 model.config.max_position_embeddings = new_max_length
 model.embeddings.position_ids.max_len = new_max_length # position_ids max_len ()


 print(f" {new_max_length},position_ids token_type_ids buffer.")




def format(item):
 #, User Agent
 anchor = ""
 if 'history' in item:
 for i, line in enumerate(item["history"][:-1]):
 if i % 2!= 0:
 anchor += "Agent: " + line
 else:
 anchor += "User: " + line

 if i < len(item["history"][:-1]) - 1:
 anchor += "\n"
 positive = item["gold_document"]
 negatives = item["document_list"]
 del negatives[int(item["local_did"][-1]) - 1]
 else:
 for i, line in enumerate(item["messages"][:-1]):
 if line['role'] =='assistant':
 anchor += "Agent: " + line['content']
 else:
 anchor += "User: " + line['content']

 if i < len(item["messages"][:-1]) - 1:
 anchor += "\n"

 positive = item["ground_truth_ctx"]['text']
 negatives = [x['text'] for x in item["ctxs"]]
 

 


 return {
 "anchor": anchor,
 "positive": positive,
 "negative": negatives
 }



def load_dataset(path):
 dataset = datasets.load_from_disk(path)
 print(dataset)

 # dataset = dataset.filter(lambda x: len(x['ctxs'])>8,num_proc=32)
 # dataset = dataset.shuffle(seed=42,keep_in_memory=True)
 dataset = dataset.select(range(26000))
 
 
 dataset = dataset.map(format, num_proc=32, remove_columns=dataset.column_names)

 
 return dataset

# ---------------------------
# 3) collate_fn: token 
#,, compute_loss 
# ---------------------------
def collate_fn(batch, query_tokenizer,context_tokenizer,max_length):
 anchors = [item["anchor"] for item in batch]
 positives = [item["positive"] for item in batch]
 negatives_list = [item["negative"] for item in batch]

 anchor_input = query_tokenizer(
 anchors,
 padding=True,
 truncation=True, # <-- Add
 max_length=max_length, # <-- Add
 return_tensors="pt",
)
 positive_input = context_tokenizer(
 positives,
 padding=True,
 truncation=True, # <-- Add
 max_length=max_length, # <-- Add
 return_tensors="pt",
)

 flattened_negatives = []
 for neg_docs in negatives_list:
 flattened_negatives.extend(neg_docs)

 
 negative_input = context_tokenizer(
 flattened_negatives,
 padding=True,
 truncation=True, 
 max_length=max_length, 
 return_tensors="pt",
)

 return {
 "anchor_input": anchor_input,
 "positive_input": positive_input,
 "negative_input": negative_input,
 }


class BiEncoderTrainer(Trainer):


 def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
 anchor_input = inputs["anchor_input"]
 positive_input = inputs["positive_input"]
 negative_input = inputs["negative_input"]

 batch_size=32
 # anchor positive
 anchor_embeds_list = []
 num_anchors = anchor_input["input_ids"].size(0)
 for i in range(0, num_anchors, batch_size):
 batch_anchor_input = {
 key: value[i:i+batch_size] for key, value in anchor_input.items()
 }
 anchor_outputs = model(**batch_anchor_input)
 anchor_embeds = anchor_outputs.last_hidden_state[:, 0]
 anchor_embeds_list.append(anchor_embeds)

 # anchor_embeds
 anchor_embeds = torch.cat(anchor_embeds_list, dim=0)

 # positive_input
 positive_embeds_list = []
 num_positives = positive_input["input_ids"].size(0)
 for i in range(0, num_positives, batch_size):
 batch_positive_input = {
 key: value[i:i+batch_size] for key, value in positive_input.items()
 }
 positive_outputs = model(**batch_positive_input)
 positive_embeds = positive_outputs.last_hidden_state[:, 0]
 positive_embeds_list.append(positive_embeds)

 # positive_embeds
 positive_embeds = torch.cat(positive_embeds_list, dim=0)
 
 # negative_outputs = model(**negative_input) 
 # negative_embeds = negative_outputs.last_hidden_state[:, 0]
 
 negative_embeds_list = []
 num_negatives = negative_input["input_ids"].size(0)
 for i in range(0, num_negatives, batch_size):
 batch_negative_input = {
 key: value[i:i+batch_size] for key, value in negative_input.items()
 }
 negative_outputs = model(**batch_negative_input)
 negative_embeds = negative_outputs.last_hidden_state[:, 0]
 negative_embeds_list.append(negative_embeds)

 # negative_embeds
 negative_embeds = torch.cat(negative_embeds_list, dim=0)

 anchors = anchor_embeds # (batch_size, embedding_dim)
 
 # 
 candidates = torch.cat([positive_embeds, negative_embeds], dim=0) # (batch_size * (1 + num_negatives), embedding_dim)


 loss = infonce_loss(anchors,candidates,temperature=0.1)
 # loss, grad_norm, fp16 NAN,.
 # 
 # F.cosine_similarity expects the input tensors to have shape (batch_size, embedding_dim)
 # scores = F.cosine_similarity(anchors.unsqueeze(1), candidates.unsqueeze(0), dim=2) *20
 # # (batch_size, batch_size * (1 + num_negatives))
 
 
 # # anchor[i] should be most similar to candidates[i], as that is the paired positive,
 # # so the label for anchor[i] is i
 # range_labels = torch.arange(0, scores.size(0), device=scores.device)
 # loss_func = nn.CrossEntropyLoss()
 # loss = loss_func(scores, range_labels)
 
 return loss
 
 def prediction_step(self, model,inputs,prediction_loss_only: bool, ignore_keys):
 with torch.inference_mode():
 anchor_input = inputs["anchor_input"]
 positive_input = inputs["positive_input"]
 negative_input = inputs["negative_input"]

 batch_size=32
 # anchor positive
 anchor_embeds_list = []
 num_anchors = anchor_input["input_ids"].size(0)
 for i in range(0, num_anchors, batch_size):
 batch_anchor_input = {
 key: value[i:i+batch_size] for key, value in anchor_input.items()
 }

 anchor_outputs = model(**batch_anchor_input)
 anchor_embeds = anchor_outputs.last_hidden_state[:, 0]
 anchor_embeds_list.append(anchor_embeds)

 # anchor_embeds
 anchor_embeds = torch.cat(anchor_embeds_list, dim=0)

 # positive_input
 positive_embeds_list = []
 num_positives = positive_input["input_ids"].size(0)
 for i in range(0, num_positives, batch_size):
 batch_positive_input = {
 key: value[i:i+batch_size] for key, value in positive_input.items()
 }
 positive_outputs = model(**batch_positive_input)
 positive_embeds = positive_outputs.last_hidden_state[:, 0]
 positive_embeds_list.append(positive_embeds)

 # positive_embeds
 positive_embeds = torch.cat(positive_embeds_list, dim=0)
 
 # negative_outputs = model(**negative_input) 
 # negative_embeds = negative_outputs.last_hidden_state[:, 0]
 
 negative_embeds_list = []
 num_negatives = negative_input["input_ids"].size(0)
 for i in range(0, num_negatives, batch_size):
 batch_negative_input = {
 key: value[i:i+batch_size] for key, value in negative_input.items()
 }
 negative_outputs = model(**batch_negative_input)
 negative_embeds = negative_outputs.last_hidden_state[:, 0]
 negative_embeds_list.append(negative_embeds)

 # negative_embeds
 negative_embeds = torch.cat(negative_embeds_list, dim=0)

 anchors = anchor_embeds # (batch_size, embedding_dim)
 
 # 
 candidates = torch.cat([positive_embeds, negative_embeds], dim=0) # (batch_size * (1 + num_negatives), embedding_dim)


 loss = infonce_loss(anchors,candidates,temperature=0.1)


 return loss,None,None

 
def parse_args():
 parser = argparse.ArgumentParser()
 parser.add_argument('--train_dataset_name',default='mtr')
 parser.add_argument('--chatrag_dataset')
 parser.add_argument('--mtr_dataset')
 parser.add_argument('--model_dir')
 parser.add_argument('--save_name')
 parser.add_argument('--output_dir')
 return parser.parse_args()


if __name__=='__main__':

 args = parse_args()

 model_path= args.model_dir

 max_length=8192
 
 query_tokenizer= AutoTokenizer.from_pretrained(model_path,truncation_side='left')
 context_tokenizer = AutoTokenizer.from_pretrained(model_path)

 if args.train_dataset_name=='mtr':
 dataset = load_dataset(args.mtr_dataset)
 
 elif args.train_dataset_name=='chatrag':
 dataset = datasets.load_from_disk(args.chatrag_dataset)
 
 else:
 assert False,"not defined dataset"


 dataset = dataset.train_test_split(test_size=0.05)
 train_ds,eval_ds = dataset['train'],dataset['test']
 print(len(train_ds),len(eval_ds))
 print(train_ds[0]['anchor'])

 model = AutoModel.from_pretrained(model_path)
 extend_position_embeddings(model,max_length)
 # 
 training_args = TrainingArguments(
 output_dir=os.path.join(args.output_dir,args.save_name),
 overwrite_output_dir=True,
 logging_dir="./logs",
 logging_steps=1,
 eval_on_start = True,
 eval_steps = 50,
 eval_strategy= 'steps',
 save_strategy='epoch',
 save_total_limit=2,
 num_train_epochs=2,
 warmup_ratio=0.05,
 learning_rate=3e-5,
 weight_decay=0.01,
 gradient_accumulation_steps=8,
 per_device_train_batch_size=32,
 per_device_eval_batch_size=32,
 dataloader_num_workers=8,
 fp16=True,
 lr_scheduler_type="cosine",
 remove_unused_columns=False,
 # gradient_checkpointing=True,
)


 trainer = BiEncoderTrainer(
 model=model,
 args=training_args,
 tokenizer=query_tokenizer,
 train_dataset=train_ds,
 eval_dataset=eval_ds,
 data_collator=lambda batch: collate_fn(batch, query_tokenizer,context_tokenizer,max_length)
)

 trainer.train()
 
