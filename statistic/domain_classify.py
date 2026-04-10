from functools import partial
import json
import multiprocessing
import time
import datasets
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig, DataCollatorWithPadding,AutoModelForSequenceClassification
from huggingface_hub import PyTorchModelHubMixin
from datasets.distributed import split_dataset_by_node
from torch.cuda import device_count
from torch.utils.data import DataLoader
from concurrent.futures import ProcessPoolExecutor
import os
from rich import progress
from loguru import logger
import numpy as np

from collections import Counter

class NvidiaClassifierModel(nn.Module, PyTorchModelHubMixin):
 def __init__(self, config):
 super(NvidiaClassifierModel, self).__init__()
 self.model = AutoModel.from_pretrained(config["base_model"],torch_dtype=torch.float16)
 
 self.dropout = nn.Dropout(config["fc_dropout"])
 self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]),dtype=torch.float16)

 def forward(self, input_ids, attention_mask):
 features = self.model(
 input_ids=input_ids, attention_mask=attention_mask
).last_hidden_state
 dropped = self.dropout(features)
 outputs = self.fc(dropped)
 return torch.softmax(outputs[:, 0,:], dim=1)



def tokenize(instance,tokenizer):

 token = tokenizer.encode(instance['text'],truncation=True, max_length=512)
 return {'input_ids': token, 'length':len(token)}



def generate_tag_by_classifier(model_path, dataset, rank, world_size, _progress, task_id):

 device = torch.device(f'cuda:{rank}')
 origin_dataset = split_dataset_by_node(dataset, rank, world_size)
 tokenizer = AutoTokenizer.from_pretrained(model_path)
 model = NvidiaClassifierModel.from_pretrained(model_path).to(device)
 model.eval()
 # if rank == 0:
 # logger.info(f'start compile model')

 # model = torch.compile(model)
 # if rank == 0:
 # logger.info(f'compile done')
 

 dataset_sorted = origin_dataset.sort('length') # reduce mapping
 # input_ids 
 dataset_processed = origin_dataset.remove_columns([col for col in origin_dataset.column_names if col!= 'input_ids'])

 collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)


 batch_size = 128 #, 128 64

 dataloader = DataLoader(dataset_processed, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=0)


 gpu_batch_results = [] 


 with torch.inference_mode():
 total_batches = len(dataloader)
 for i, batch in enumerate(dataloader):
 batch = batch.to(device)

 
 outputs = model(**batch)
 # ( GPU) 
 gpu_batch_results.append(torch.argmax(outputs, dim=1).cpu())
 _progress[task_id] = {"progress": i + 1, "total": total_batches}
 
 return gpu_batch_results



def classify_document_domain(dataset, model_path,domain_result_path,worker_func=generate_tag_by_classifier):

 config = AutoConfig.from_pretrained(model_path)
 tokenizer = AutoTokenizer.from_pretrained(model_path)
 dataset = dataset.map(partial(tokenize, tokenizer=tokenizer), num_proc=64,load_from_cache_file=False)
 dataset = dataset.shuffle().select(range(100000))
 num_processes = device_count()
 args_list = [(model_path, dataset, rank, num_processes) for rank in range(num_processes)]

 with progress.Progress(
 "[progress.description]{task.description}",
 progress.BarColumn(),
 "[progress.percentage]{task.percentage:>3.0f}%",
 progress.TimeRemainingColumn(),
 progress.TimeElapsedColumn(),
 refresh_per_second=1,
) as progress_bar:
 futures = []

 with multiprocessing.Manager() as manager:
 _progress = manager.dict()
 overall_progress_task = progress_bar.add_task("[green]All jobs progress:")

 with ProcessPoolExecutor() as executor:
 for i in range(num_processes):
 task_id = progress_bar.add_task(f"task {i}", visible=False)
 func = partial(worker_func, _progress=_progress,task_id = task_id)
 futures.append(executor.submit(func, *args_list[i]))

 
 while (n_finished:= sum(future.done() for future in futures)) < len(futures):
 progress_bar.update(
 overall_progress_task, completed=n_finished, total=len(futures)
)
 for task_id, update_data in _progress.items():
 latest = update_data["progress"]
 total = update_data["total"]
 progress_bar.update(
 task_id,
 completed=latest,
 total=total,
 visible=latest < total,
)
 time.sleep(0.1)

 # 
 results = []
 for future in futures:
 results.extend(future.result())

 results = torch.cat(results, dim=0).tolist()
 
 predicted_domains = Counter([config.id2label[class_idx] for class_idx in results])
 print('done')
 # 
 json.dump(predicted_domains,open(os.path.join(domain_result_path,'domain_classify_result.json'),'w'),ensure_ascii=False, indent=4)
 return predicted_domains


 
 








