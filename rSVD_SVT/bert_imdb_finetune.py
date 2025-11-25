import sys
import os
import gc

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from custom_adam.optimizer import CustomAdam
from rsvd_svt import RandomizedSVDGradientProjector
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
from rSVD.utils import (
    get_device, reset_memory_stats, get_peak_memory_mb,
    get_current_memory_mb, extract_timing_stats, 
    get_profiler_activities)
from torch.profiler import profile


device = get_device()

# Model, tokenizer, and dataset
model_name = "bert-base-uncased"
max_length = 256
batch_size = 32
num_epochs = 3

# ============================================================
# Load Dataset
# ============================================================
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

train_loader = DataLoader(
    tokenized_datasets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
    pin_memory=False,  
    num_workers=0, 
)

test_loader = DataLoader(
    tokenized_datasets["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=False,
    num_workers=0,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)

optimizer = CustomAdam(params = model.parameters())

def train_epoch(model, loader, optimizer, device, rsvd_projector):
    """Train for one epoch."""
    reset_memory_stats(device)

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    profile_results = None
    activities = get_profiler_activities(device)

    prof = None
    with profile(
        activities=activities,
        profile_memory=True,
    ) as prof:
        progress_bar = tqdm(loader, desc="Training")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
        
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits = outputs.logits
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            rsvd_projector.step(model)
            optimizer.step()
        
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
        
            progress_bar.set_postfix({
                'loss': f'{total_loss / (step + 1):.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    accuracy = correct / total

    peak_mem = get_peak_memory_mb(device)
    if peak_mem is not None:
        profile_results = {
            'peak_memory_mb': peak_mem,
            'profiler': prof
        }
    else:
        profile_results = {'profiler': prof}

    return total_loss, accuracy, profile_results


def evaluate(model, loader, device):
    """Evaluate the model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    accuracy = correct / total
    return accuracy

train_losses = []
train_accs = []
val_accs = []

best_val_acc = 0.0

initial_memory = get_current_memory_mb(device)
if initial_memory is not None:
    print(f"Initial GPU memory: {initial_memory:.2f} MB")

# Tune This: 64, 128, 256
ranks = [64]


for rank in ranks:
    rsvd_projector = RandomizedSVDGradientProjector(
        rank=rank, 
    )   
    for epoch in range(1, num_epochs + 1): 
        # Train
        train_loss, train_acc, profile_results = train_epoch(
            model, train_loader, optimizer, device, rsvd_projector
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
    
        # Evaluate
        val_acc = evaluate(model, test_loader, device)
        val_accs.append(val_acc)
    
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Acc:   {val_acc*100:.2f}%")

        if profile_results:
            peak_memory = profile_results.get('peak_memory_mb', 0)
            print(f"Memory Usage (Peak): {peak_memory:.3f}")   

        timing_stats = extract_timing_stats(
            profile_results['profiler'] if profile_results else None, device, "Profiler Results") 

        if device.startswith("cuda") and torch.cuda.is_available():
            if 'cuda_time_ms' in timing_stats:
                cuda_time = timing_stats['cuda_time_ms']
            print(f"Overall Epoch CUDA Time: {cuda_time:.3f}")
    
        if 'cpu_time_ms' in timing_stats:
            cpu_time = timing_stats['cpu_time_ms'] 
            print(f"Overall Epoch CPU Time: {cpu_time:.3f}")
        
        del profile_results
        gc.collect()
        torch.cuda.empty_cache()
