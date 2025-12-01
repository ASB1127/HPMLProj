"""
Fine-tune DistilBERT on IMDB or SST-2 Sentiment Classification using rSVDAdam optimizer.
Usage: python bert_finetune.py --dataset imdb
       python bert_finetune.py --dataset sst
"""
import argparse
import torch
import csv
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
import numpy as np
import os
from torch.profiler import profile, ProfilerActivity
import gc
from optimizer import rSVDAdam
from utils import (
    get_device, reset_memory_stats, get_peak_memory_mb,
    get_current_memory_mb, count_parameters
)

# ============================================================
# Parse Arguments
# ============================================================
parser = argparse.ArgumentParser(description='Fine-tune BERT on IMDB or SST-2 with rSVDAdam')
parser.add_argument(
    '--dataset',
    type=str,
    choices=['imdb', 'sst'],
    required=True,
    help='Dataset to use: "imdb" or "sst"'
)
args = parser.parse_args()

dataset_name = args.dataset.lower()

# ============================================================
# Dataset-Specific Configuration
# ============================================================
if dataset_name == 'imdb':
    dataset_config = {
        'name': 'imdb',
        'dataset_path': 'imdb',
        'text_column': 'text',
        'train_split': 'train',
        'eval_split': 'test',
        'remove_columns': ['text'],
        'save_dir': './bert_imdb_checkpoints',
        'model_name': 'distilbert-base-uncased',
        'max_length': 128,
        'batch_size': 32,
        'eval_batch_size': 64,
        'learning_rate': 2e-4,
    }
elif dataset_name == 'sst':
    dataset_config = {
        'name': 'sst2',
        'dataset_path': 'glue',
        'dataset_config': 'sst2',
        'text_column': 'sentence',
        'train_split': 'train',
        'eval_split': 'validation',
        'remove_columns': ['sentence', 'idx'],
        'save_dir': './bert_sst2_checkpoints',
        'model_name': 'distilbert-base-uncased',
        'max_length': 128,
        'batch_size': 32,
        'eval_batch_size': 64,
        'learning_rate': 2e-4,
    }
else:
    raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'imdb' or 'sst'")

# ============================================================
# Configuration
# ============================================================
device = get_device()

# Model and dataset configs
model_name = dataset_config['model_name']
max_length = dataset_config['max_length']
batch_size = dataset_config['batch_size']
gradient_accumulation_steps = 4 

# Training hyperparameters
num_epochs = 10
learning_rate = dataset_config['learning_rate']
weight_decay = 0.01

# rSVDAdam specific parameters
rank_fraction = 0.3 
proj_interval = 500  # Recompute projector every 500 steps
use_rgp = True

# Profiling options
enable_training_profiling = True  # Set to False to disable per-epoch profiling (faster training)
profile_steps_per_epoch = 10  # Number of steps to profile per epoch (reduces overhead)

# Evaluation
eval_batch_size = dataset_config['eval_batch_size']
save_dir = dataset_config['save_dir']
os.makedirs(save_dir, exist_ok=True)

print(f"Using device: {device}")
print(f"Dataset: {dataset_config['name'].upper()}")
print(f"Model: {model_name}")
print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
print(f"Output directory: {save_dir}")

# ============================================================
# Load Dataset
# ============================================================
print(f"\nLoading {dataset_config['name'].upper()} dataset...")

if dataset_name == 'imdb':
    dataset = load_dataset(dataset_config['dataset_path'])
else:  # sst
    dataset = load_dataset(dataset_config['dataset_path'], dataset_config['dataset_config'])

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    """Tokenize the examples."""
    return tokenizer(
        examples[dataset_config['text_column']],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

print("Tokenizing dataset...")
tokenized_train = dataset[dataset_config['train_split']].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset_config['remove_columns']
)
tokenized_eval = dataset[dataset_config['eval_split']].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset_config['remove_columns']
)

print(f"Train samples: {len(tokenized_train)}")
print(f"Eval samples: {len(tokenized_eval)}")

# ============================================================
# Load Model
# ============================================================
print(f"\nLoading {model_name}...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)

# Count parameters
param_counts = count_parameters(model)
print(f"Total parameters: {param_counts['total']:,}")
print(f"Trainable parameters: {param_counts['trainable']:,}")

# ============================================================
# Convert to PyTorch format for Trainer compatibility
# ============================================================
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ============================================================
# Trainer Setup
# ============================================================

def compute_metrics(eval_pred):
    """Compute accuracy from logits and labels."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}


class ProfilerCallback(TrainerCallback):
    """Callback to profile CUDA memory and time for each epoch during training."""
    
    def __init__(self, save_dir, profile_steps=50):
        self.profiler = None
        self.profile_steps = profile_steps
        self.epoch_num = 0
        self.step_in_epoch = 0
        self.profiling_active = False
        self.save_dir = save_dir
        
    def on_epoch_begin(self, args, state, control, **kwargs): 
        self.step_in_epoch = 0
        self.profiling_active = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=True,
                with_stack=False
            )
            self.profiler.__enter__()
            self.profiling_active = True
            print(f"\n=== Started profiling Epoch {self.epoch_num + 1} (first {self.profile_steps} steps) ===")

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiling_active:
            self.step_in_epoch += 1

            if self.step_in_epoch >= self.profile_steps:
                self.profiler.__exit__(None, None, None)
                self.profiling_active = False

                epoch_label = self.epoch_num + 1

                # Save CUDA time profile
                profile_time_file = os.path.join(self.save_dir, f"profile_cuda_time_epoch_{epoch_label}.txt")
                with open(profile_time_file, "w") as f:
                    f.write(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=200))
                
                # Save CUDA memory profile
                profile_memory_file = os.path.join(self.save_dir, f"profile_cuda_memory_epoch_{epoch_label}.txt")
                with open(profile_memory_file, "w") as f:
                    f.write(self.profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=200))
                
                print(f"\n=== Epoch {epoch_label} Profile Saved ===")
                print("\n=== TOP 10 TIME (CUDA) ===")
                print(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

                print("\n=== TOP 10 GPU MEMORY ===")
                print(self.profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

                # Export Chrome trace
                trace_file = os.path.join(self.save_dir, f"trace_epoch_{epoch_label}.json")
                self.profiler.export_chrome_trace(trace_file)

                self.profiler = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.profiling_active and self.profiler:
            self.profiler.__exit__(None, None, None)
            self.profiling_active = False

            self.profiler = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.epoch_num += 1


class GPUTimeTracker(TrainerCallback):
    """Callback to track GPU time per epoch."""
    
    def __init__(self, save_dir):
        self.epoch_times = []
        self.epoch_start_time = None
        self.save_dir = save_dir
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.epoch_start_time = torch.cuda.Event(enable_timing=True)
            self.epoch_start_time.record()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and self.epoch_start_time is not None:
            epoch_end_time = torch.cuda.Event(enable_timing=True)
            epoch_end_time.record()
            torch.cuda.synchronize()
            
            elapsed_time_ms = self.epoch_start_time.elapsed_time(epoch_end_time)
            self.epoch_times.append({
                "epoch": int(state.epoch),
                "gpu_time_ms": elapsed_time_ms,
            })
            print(f"Epoch {int(state.epoch)} GPU time: {elapsed_time_ms:.2f} ms")
    
    def save_csv(self, filename=None):
        """Write GPU time stats to a CSV file."""
        if filename is None:
            filename = os.path.join(self.save_dir, "epoch_gpu_time.csv")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "gpu_time_ms"])

            for stat in self.epoch_times:
                writer.writerow([
                    stat["epoch"],
                    stat["gpu_time_ms"]
                ])


class LossTrackerCallback(TrainerCallback):
    """Callback to track and save loss per epoch."""
    
    def __init__(self, save_dir):
        self.epoch_losses = []
        self.save_dir = save_dir
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture loss from training logs."""
        if logs is not None:
            epoch = int(state.epoch)
            train_loss = logs.get('loss', None)
            eval_loss = logs.get('eval_loss', None)
            
            # Find or create entry for this epoch
            existing = next((x for x in self.epoch_losses if x['epoch'] == epoch), None)
            if existing:
                if train_loss is not None:
                    existing['train_loss'] = train_loss
                if eval_loss is not None:
                    existing['eval_loss'] = eval_loss
            else:
                self.epoch_losses.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                })
    
    def save_csv(self, filename=None):
        """Write loss stats to a CSV file."""
        if filename is None:
            filename = os.path.join(self.save_dir, "epoch_loss.csv")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "eval_loss"])
            
            for stat in self.epoch_losses:
                writer.writerow([
                    stat["epoch"],
                    stat.get("train_loss", ""),
                    stat.get("eval_loss", "")
                ])


class MemoryMonitorCallback(TrainerCallback):
    """Callback to monitor and save memory usage per epoch."""
    
    def __init__(self, save_dir):
        self.epoch_stats = []
        self.save_dir = save_dir
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            current = torch.cuda.memory_allocated()
            
            self.epoch_stats.append({
                "epoch": int(state.epoch),
                "current_allocated": current,
                "peak_allocated": peak,
            })
    
    def save_csv(self, filename=None):
        """Write epoch_stats to a CSV file."""
        if filename is None:
            filename = os.path.join(self.save_dir, "epoch_memory.csv")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "current_allocated_bytes", "peak_allocated_bytes"])

            for stat in self.epoch_stats:
                writer.writerow([
                    stat["epoch"],
                    stat["current_allocated"],
                    stat["peak_allocated"]
                ])


class RsvdTrainer(Trainer):
    """Custom Trainer that uses rSVDAdam."""

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            self.optimizer = rSVDAdam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                rank_fraction=rank_fraction,
                proj_interval=proj_interval,
                use_rgp=use_rgp,
                weight_decay=self.args.weight_decay,
                decoupled_weight_decay=True,
                verbose_memory_once=True,
            )

        if self.lr_scheduler is None:
            # Constant learning rate scheduler (no warmup)
            self.lr_scheduler = LambdaLR(self.optimizer, lambda _: 1.0)


training_args = TrainingArguments(
    output_dir=save_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    eval_strategy="epoch",  
    save_strategy="epoch",  
    logging_dir=os.path.join(save_dir, "logs"),
    logging_steps=50,
    save_total_limit=3,
)

print("\n" + "=" * 70)
print(f"Starting Training with Hugging Face Trainer - {dataset_config['name'].upper()}")
print("=" * 70)

# Optional memory tracking before training
reset_memory_stats(device)
initial_memory = get_current_memory_mb(device)
if initial_memory is not None:
    print(f"Initial GPU memory: {initial_memory:.2f} MB")

# Initialize callbacks
loss_tracker = LossTrackerCallback(save_dir)
memory_callback = MemoryMonitorCallback(save_dir)
gpu_time_tracker = GPUTimeTracker(save_dir)

callbacks = [
    loss_tracker,
    memory_callback,
    gpu_time_tracker,
]

# Add profiling callback only if enabled
if enable_training_profiling:
    profiler_callback = ProfilerCallback(save_dir, profile_steps=profile_steps_per_epoch)
    callbacks.append(profiler_callback)
    print(f"Training profiling enabled: profiling first {profile_steps_per_epoch} steps per epoch")
else:
    print("Training profiling disabled (pre-training profiling still active)")

trainer = RsvdTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

# Profile optimizer step only (before main training)
print("\n" + "=" * 70)
print("Profiling Optimizer Step")
print("=" * 70)

# Create optimizer for profiling (same as trainer will use)
optimizer = rSVDAdam(
    model.parameters(),
    lr=learning_rate,
    rank_fraction=rank_fraction,
    proj_interval=proj_interval,
    use_rgp=use_rgp,
    weight_decay=weight_decay,
    decoupled_weight_decay=True,
    verbose_memory_once=True,
)

# Get dataloader for profiling
train_dataloader = trainer.get_train_dataloader()

# Warmup: run a few steps without profiling (get new batch each time, like Lora)
print("Warmup steps...")
model.train()
for _ in range(2):
    batch = next(iter(train_dataloader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    optimizer.zero_grad()
    out = model(**batch)
    out.loss.backward()
    optimizer.step()
    del batch, out
    torch.cuda.empty_cache()
    gc.collect()

torch.cuda.synchronize()

# Profile only the optimizer step (get a fresh batch)
print("Profiling optimizer.step()...")
batch = next(iter(train_dataloader))
batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
         for k, v in batch.items()}
optimizer.zero_grad()
out = model(**batch)
out.loss.backward()

# Profile only optimizer step
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    optimizer.step()

# Save profiler results to dataset-specific directory
profile_time_file = os.path.join(save_dir, "profile_cuda_time.txt")
profile_memory_file = os.path.join(save_dir, "profile_cuda_memory.txt")
profile_trace_file = os.path.join(save_dir, "profile_trace.json")

print("\n=== TOP 20 TIME (CUDA) ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

print("\n=== TOP 20 GPU MEMORY ===")
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

with open(profile_time_file, "w") as f:
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=200))

with open(profile_memory_file, "w") as f:
    f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=200))

# Export Chrome trace
prof.export_chrome_trace(profile_trace_file)
print(f"Profiler output saved:")
print(f"  - {profile_time_file}")
print(f"  - {profile_memory_file}")
print(f"  - {profile_trace_file}")

# Cleanup profiling optimizer and gradients
optimizer.zero_grad()
del optimizer, batch, out
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Now start main training (trainer will create its own optimizer)
print("\n" + "=" * 70)
print("Starting Main Training")
print("=" * 70)

train_result = trainer.train()
trainer.save_model(save_dir)

metrics = trainer.evaluate()
print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
print(f"Best Validation Accuracy (from Trainer): {metrics.get('eval_accuracy', 0.0) * 100:.2f}%")
print(f"Final Validation Loss: {metrics.get('eval_loss', float('nan')):.4f}")

peak_memory = get_peak_memory_mb(device)
if peak_memory is not None:
    print(f"Peak GPU Memory Usage: {peak_memory:.2f} MB")

# Save loss tracking CSV
loss_tracker.save_csv()
print(f"Saved epoch loss stats → {os.path.join(save_dir, 'epoch_loss.csv')}")

# Save memory monitoring CSV
memory_callback.save_csv()
print(f"Saved epoch memory stats → {os.path.join(save_dir, 'epoch_memory.csv')}")

# Save GPU time tracking CSV
gpu_time_tracker.save_csv()
print(f"Saved GPU time stats → {os.path.join(save_dir, 'epoch_gpu_time.csv')}")

print(f"\nAll results saved to: {save_dir}")

