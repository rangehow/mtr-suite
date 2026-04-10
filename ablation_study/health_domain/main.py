import os
import json
import time
import multiprocessing
from functools import partial

import datasets
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda import device_count
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig, DataCollatorWithPadding
from huggingface_hub import PyTorchModelHubMixin
from datasets.distributed import split_dataset_by_node
from rich import progress
from loguru import logger
from collections import Counter


class NvidiaClassifierModel(nn.Module, PyTorchModelHubMixin):
 def __init__(self, config):
 super(NvidiaClassifierModel, self).__init__()
 self.model = AutoModel.from_pretrained(config["base_model"], torch_dtype=torch.float16)
 self.dropout = nn.Dropout(config["fc_dropout"])
 self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]), dtype=torch.float16)

 def forward(self, input_ids, attention_mask):
 features = self.model(
 input_ids=input_ids, attention_mask=attention_mask
).last_hidden_state
 dropped = self.dropout(features)
 outputs = self.fc(dropped)
 return torch.softmax(outputs[:, 0,:], dim=1)

def tokenize(instance, tokenizer):
 # 'text' 
 #, 'text'
 token = tokenizer.encode(instance['text'], truncation=True, max_length=512)
 return {'input_ids': token, 'length': len(token)}

def generate_tag_by_classifier(model_path, dataset_slice, rank, world_size, _progress, task_id):
 """
, GPU.
:.
 """
 device = torch.device(f'cuda:{rank}')
 tokenizer = AutoTokenizer.from_pretrained(model_path)
 model = NvidiaClassifierModel.from_pretrained(model_path).to(device)
 model.eval()
 
 dataset_processed = dataset_slice.remove_columns([col for col in dataset_slice.column_names if col not in ['input_ids', 'attention_mask']])
 collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
 
 # batch_size
 batch_size = 128
 dataloader = DataLoader(dataset_processed, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=0)

 gpu_batch_results = []
 with torch.inference_mode():
 total_batches = len(dataloader)
 for i, batch in enumerate(dataloader):
 batch = {k: v.to(device) for k, v in batch.items()}
 outputs = model(**batch)
 gpu_batch_results.append(torch.argmax(outputs, dim=1).cpu())
 if _progress:
 _progress[task_id] = {"progress": i + 1, "total": total_batches}
 
 # Tensor
 return torch.cat(gpu_batch_results)



def classify_and_filter_dataset(
 dataset_path: str, 
 model_path: str, 
 output_path: str,
 target_domain: str = "Health"
):
 """
:,,.
 """
 logger.info("...")
 

 logger.info(f" {model_path}...")
 config = AutoConfig.from_pretrained(model_path)
 tokenizer = AutoTokenizer.from_pretrained(model_path)


 logger.info(f" {dataset_path}...")
 dataset = datasets.load_from_disk(dataset_path)
 
 logger.info("...")
 # 
 tokenized_dataset = dataset.map(
 partial(tokenize, tokenizer=tokenizer), 
 num_proc=max(1, os.cpu_count() // 2), 
 load_from_cache_file=False
)
 
 # 
 tokenized_dataset = tokenized_dataset.remove_columns([c for c in dataset.column_names if c!= "text"])

 logger.info("...")
 sorted_dataset = tokenized_dataset.sort('length')

 # 3. GPU 
 num_processes = device_count()
 if num_processes == 0:
 raise RuntimeError(" CUDA,.")
 logger.info(f" {num_processes} GPU,...")

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
 overall_progress_task = progress_bar.add_task("[green] GPU:", total=num_processes)
 
 with ProcessPoolExecutor(max_workers=num_processes) as executor:
 for i in range(num_processes):
 # GPU 
 dataset_slice = split_dataset_by_node(sorted_dataset, rank=i, world_size=num_processes)
 task_id = progress_bar.add_task(f"[cyan]GPU {i}", visible=False)
 #: dataset_slice
 future = executor.submit(
 generate_tag_by_classifier, 
 model_path, 
 dataset_slice, 
 i, 
 num_processes, 
 _progress, 
 task_id
)
 futures.append(future)

 while (n_finished:= sum(future.done() for future in futures)) < len(futures):
 progress_bar.update(overall_progress_task, completed=n_finished)
 for task_id, update_data in _progress.items():
 latest, total = update_data["progress"], update_data["total"]
 progress_bar.update(task_id, completed=latest, total=total, visible=latest < total)
 time.sleep(0.1)
 
 progress_bar.update(overall_progress_task, completed=len(futures), description="[bold green] ✓")


 logger.info(" GPU...")
 results_from_gpus = [future.result() for future in futures]
 all_predictions = torch.cat(results_from_gpus, dim=0).tolist()
 
 # 
 if len(all_predictions)!= len(sorted_dataset):
 logger.error(f" ({len(all_predictions)}) ({len(sorted_dataset)})!")
 return

 logger.info(f" {len(all_predictions)}.")


 logger.info("...")
 classified_dataset = sorted_dataset.add_column("predicted_label_id", all_predictions)
 
 predicted_domains = Counter(config.id2label[class_idx] for class_idx in all_predictions)
 logger.info(f": {json.dumps(dict(predicted_domains), indent=2, ensure_ascii=False)}")
 
 # "Health" ID
 if target_domain not in config.label2id:
 logger.error(f" '{target_domain}'.: {list(config.label2id.keys())}")
 return
 
 health_label_id = config.label2id[target_domain]
 logger.info(f" '{target_domain}' ID: {health_label_id}")
 
 logger.info(f" '{target_domain}'...")
 health_subset = classified_dataset.filter(
 lambda example: example['predicted_label_id'] == health_label_id,
 num_proc=max(1, os.cpu_count() // 2) # 
)
 
 logger.info(f", {len(health_subset)} '{target_domain}'.")


 logger.info(f": {output_path}")
 health_subset.save_to_disk(output_path)
 logger.info(f"[bold green]! {output_path}[/bold green]")


if __name__ == '__main__':

 CLASSIFIER_MODEL_PATH = "nvidia/domain-classifier"
 
 # 2. Hugging Face 
 INPUT_DATASET_PATH = "/path/to/mtr/mtr-data-dumps/processed_dataset"
 
 OUTPUT_HEALTH_SUBSET_PATH = "/path/to/mtr/mtr-data-dumps/health_document_subset"

 classify_and_filter_dataset(
 dataset_path=INPUT_DATASET_PATH,
 model_path=CLASSIFIER_MODEL_PATH,
 output_path=OUTPUT_HEALTH_SUBSET_PATH,
 target_domain="Health"
)