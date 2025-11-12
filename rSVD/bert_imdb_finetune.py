"""
Fine-tune BERT-base-uncased on IMDB Sentiment Classification using rSVDAdam optimizer.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
from optimizer import rSVDAdam
from utils import (
    get_device, reset_memory_stats, get_peak_memory_mb,
    get_current_memory_mb, print_memory_status, count_parameters
)

# ============================================================
# Configuration
# ============================================================
device = get_device()

# Model and dataset
model_name = "bert-base-uncased"
max_length = 512
batch_size = 16  # Adjust based on GPU memory
gradient_accumulation_steps = 4  # Effective batch size = batch_size * gradient_accumulation_steps

# Training hyperparameters
num_epochs = 3
learning_rate = 2e-5
warmup_steps = 500
weight_decay = 0.01

# rSVDAdam specific parameters
rank_fraction = 0.3  # 30% of min(m, n) for rank
proj_interval = 500  # Recompute projector every 500 steps
use_rgp = True

# Evaluation
eval_batch_size = 32
save_dir = "./bert_imdb_checkpoints"
os.makedirs(save_dir, exist_ok=True)

print(f"Using device: {device}")
print(f"Model: {model_name}")
print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")

# ============================================================
# Load Dataset
# ============================================================
print("\nLoading IMDB dataset...")
dataset = load_dataset("imdb")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    """Tokenize the examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

print("Tokenizing dataset...")
tokenized_train = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
tokenized_test = dataset["test"].map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Convert to PyTorch format
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Data loaders
train_loader = DataLoader(
    tokenized_train,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True if device.startswith("cuda") else False
)

test_loader = DataLoader(
    tokenized_test,
    batch_size=eval_batch_size,
    shuffle=False,
    pin_memory=True if device.startswith("cuda") else False
)

print(f"Train samples: {len(tokenized_train)}")
print(f"Test samples: {len(tokenized_test)}")

# ============================================================
# Load Model
# ============================================================
print(f"\nLoading {model_name}...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # Positive/Negative
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)

# Count parameters
param_counts = count_parameters(model)
print(f"Total parameters: {param_counts['total']:,}")
print(f"Trainable parameters: {param_counts['trainable']:,}")

# ============================================================
# Setup Optimizer and Scheduler
# ============================================================
# Calculate total training steps
num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
print(f"\nTotal training steps: {num_training_steps}")

# Optimizer - using rSVDAdam
optimizer = rSVDAdam(
    model.parameters(),
    lr=learning_rate,
    rank_fraction=rank_fraction,
    proj_interval=proj_interval,
    use_rgp=use_rgp,
    weight_decay=weight_decay,
    decoupled_weight_decay=True,  # AdamW-style weight decay
    verbose_memory_once=True
)

# Learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

# Loss function
criterion = nn.CrossEntropyLoss()

# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, criterion, device, gradient_accumulation_steps):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(loader, desc="Training")
    for step, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Metrics
        total_loss += loss.item() * gradient_accumulation_steps
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{total_loss / (step + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating")
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss / (total // batch_size):.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================
# Training Loop
# ============================================================
print("\n" + "=" * 70)
print("Starting Training")
print("=" * 70)

# Track metrics
train_losses = []
train_accs = []
val_losses = []
val_accs = []

# Memory tracking
reset_memory_stats(device)
initial_memory = get_current_memory_mb(device)
if initial_memory is not None:
    print(f"Initial GPU memory: {initial_memory:.2f} MB")

best_val_acc = 0.0

for epoch in range(1, num_epochs + 1):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{num_epochs}")
    print(f"{'='*70}")
    
    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler,
        criterion, device, gradient_accumulation_steps
    )
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Evaluate
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Print epoch summary
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(save_dir, f"best_model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, save_path)
        print(f"  ✓ Saved best model (val_acc: {val_acc*100:.2f}%) to {save_path}")
    
    # Memory usage
    print_memory_status(device, f"Epoch {epoch}")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"Final Validation Accuracy: {val_accs[-1]*100:.2f}%")
print(f"Final Validation Loss: {val_losses[-1]:.4f}")

peak_memory = get_peak_memory_mb(device)
if peak_memory is not None:
    print(f"Peak GPU Memory Usage: {peak_memory:.2f} MB")

print(f"\nModel checkpoints saved to: {save_dir}")

