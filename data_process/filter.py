from functools import partial
import multiprocessing
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

from rich import progress
from loguru import logger

def _remove_useless_part_in_wiki(examples):
 cleaned=[]
 for example in examples['text']:
 example=example.split('\n\nSee also')[0].split('\n\nReferences')[0].split('\n\nExternal links')[0].split('\n\nNotes and references')[0].split('\n\nNotes')[0].split('\n\nFurther reading')[0]
 cleaned.append(example)

 return {"text":cleaned}



class QualityModel(nn.Module, PyTorchModelHubMixin):
 def __init__(self, config):
 super(QualityModel, self).__init__()
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


class NvidiaFineWebEDUModel(nn.Module):
 def __init__(self, model_dir):
 super(NvidiaFineWebEDUModel, self).__init__()
 self.model = AutoModelForSequenceClassification.from_pretrained(
 model_dir,
 torch_dtype=torch.bfloat16,
)
 

 def forward(self, input_ids, attention_mask):
 outputs = self.model(
 input_ids=input_ids, attention_mask=attention_mask
)

 return outputs

def tokenize(instance,tokenizer):

 token = tokenizer.encode(instance['text'],truncation=True, max_length=512)
 return {'input_ids': token, 'length':len(token)}



def generate_tag_by_classifier(model_path, dataset, rank, world_size, _progress, task_id):

 device = torch.device(f'cuda:{rank}')
 origin_dataset = split_dataset_by_node(dataset, rank, world_size)
 tokenizer = AutoTokenizer.from_pretrained(model_path)
 model = QualityModel.from_pretrained(model_path).to(device)
 model.eval()
 if rank == 0:
 logger.info(f'start compile model')

 model = torch.compile(model)
 if rank == 0:
 logger.info(f'compile done')
 

 dataset_sorted = origin_dataset.sort('length') # reduce mapping
 # input_ids 
 dataset_processed = origin_dataset.remove_columns([col for col in origin_dataset.column_names if col!= 'input_ids'])

 collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

 # batch_size 
 batch_size = 128 #, 128 64

 dataloader = DataLoader(dataset_processed, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=8,prefetch_factor=3)

 all_predicted_classes_cpu = [] # CPU 
 gpu_batch_results = [] # GPU 
 batches_processed_since_clear = 0
 batches_to_accumulate = 200 

 with torch.inference_mode():
 total_batches = len(dataloader)
 for i, batch in enumerate(dataloader):
 batch = batch.to(device)

 
 outputs = model(**batch)
 # ( GPU) 
 gpu_batch_results.append(torch.argmax(outputs, dim=1))
 batches_processed_since_clear += 1

 # batches_to_accumulate, 
 if batches_processed_since_clear >= batches_to_accumulate or (i + 1) == total_batches:
 if gpu_batch_results: 
 concatenated_gpu_results = torch.concatenate(gpu_batch_results, dim=0)
 all_predicted_classes_cpu.extend(concatenated_gpu_results.cpu().tolist())
 gpu_batch_results = []
 batches_processed_since_clear = 0
 

 _progress[task_id] = {"progress": i + 1, "total": total_batches}

 if gpu_batch_results: 
 concatenated_gpu_results = torch.concatenate(gpu_batch_results, dim=0)
 all_predicted_classes_cpu.extend(concatenated_gpu_results.cpu().tolist())
 gpu_batch_results = []
 batches_processed_since_clear = 0



 # 
 #:add_column 
 if len(all_predicted_classes_cpu)!= len(origin_dataset):
 raise ValueError(f"Length mismatch: Predictions ({len(all_predicted_classes_cpu)}) vs Dataset ({len(origin_dataset)})")

 origin_dataset = origin_dataset.add_column(name='naive_quality', column=all_predicted_classes_cpu)

 # filtered_dataset = origin_dataset.filter(lambda x: x['quality'] == 0)
 # 
 columns_to_remove = ['input_ids']
 if 'length' in origin_dataset.column_names:
 columns_to_remove.append('length')
 final_dataset = origin_dataset.remove_columns(columns_to_remove)

 # if rank == 0: #, 
 # print("Dataset after processing and filtering:")
 # print(final_dataset)
 # print(f"Original dataset size: {len(origin_dataset)}")
 # print(f"Filtered dataset size: {len(final_dataset)}")

 return final_dataset





def generate_tag_by_nvidia_finewebedu_classifier(model_path, dataset, rank, world_size, _progress, task_id):

 device = torch.device(f'cuda:{rank}')
 origin_dataset = split_dataset_by_node(dataset, rank, world_size)
 tokenizer = AutoTokenizer.from_pretrained(model_path)
 model = NvidiaFineWebEDUModel(model_path).to(device)
 model.eval()
 if rank == 0:
 logger.info(f'start compile model')
 model = torch.compile(model)
 if rank == 0:
 logger.info(f'compile done')
 

 dataset_sorted = origin_dataset.sort('length') # reduce mapping
 # input_ids 
 dataset_processed = origin_dataset.remove_columns([col for col in origin_dataset.column_names if col!= 'input_ids'])

 collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

 # batch_size 
 batch_size = 128 #, 128 64

 dataloader = DataLoader(dataset_processed, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=8,prefetch_factor=3)

 all_predicted_classes_cpu = [] # CPU 
 gpu_batch_results = [] # GPU 
 batches_processed_since_clear = 0
 batches_to_accumulate = 200 

 with torch.inference_mode():
 total_batches = len(dataloader)
 for i, batch in enumerate(dataloader):
 batch = batch.to(device)

 
 outputs = model(**batch).logits[:, 0]
 
 # ( GPU) 
 gpu_batch_results.append(outputs)
 batches_processed_since_clear += 1

 # batches_to_accumulate, 
 if batches_processed_since_clear >= batches_to_accumulate or (i + 1) == total_batches:
 if gpu_batch_results: 
 concatenated_gpu_results = torch.cat(gpu_batch_results, dim=0)
 all_predicted_classes_cpu.extend(concatenated_gpu_results.cpu().tolist())
 gpu_batch_results = []
 batches_processed_since_clear = 0
 

 _progress[task_id] = {"progress": i + 1, "total": total_batches}

 if gpu_batch_results: 
 concatenated_gpu_results = torch.concatenate(gpu_batch_results, dim=0)
 all_predicted_classes_cpu.extend(concatenated_gpu_results.cpu().tolist())
 gpu_batch_results = []
 batches_processed_since_clear = 0



 # 
 #:add_column 
 if len(all_predicted_classes_cpu)!= len(origin_dataset):
 raise ValueError(f"Length mismatch: Predictions ({len(all_predicted_classes_cpu)}) vs Dataset ({len(origin_dataset)})")

 origin_dataset = origin_dataset.add_column(name='edu_quality', column=all_predicted_classes_cpu)

 # filtered_dataset = origin_dataset.filter(lambda x: x['quality'] == 0)
 # 
 columns_to_remove = ['input_ids']
 if 'length' in origin_dataset.column_names:
 columns_to_remove.append('length')
 final_dataset = origin_dataset.remove_columns(columns_to_remove)


 return final_dataset












def filter_by_classifier(dataset, model_path,worker_func):
 tokenizer = AutoTokenizer.from_pretrained(model_path)
 dataset = dataset.map(partial(tokenize, tokenizer=tokenizer), num_proc=64,load_from_cache_file=False)
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

 # 
 results = []
 for future in futures:
 results.append(future.result())
 
 # 
 filtered_dataset = datasets.concatenate_datasets(results)
 return filtered_dataset


 
 








