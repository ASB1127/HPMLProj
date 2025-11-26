"""
Fine-tune BERT-base-uncased on SST-2 Sentiment Classification using rSVDAdam optimizer.
"""
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np
import os
from optimizer import rSVDAdam
from utils import (
    get_device, reset_memory_stats, get_peak_memory_mb,
    get_current_memory_mb, count_parameters
)

# ============================================================
# Configuration
# ============================================================
device = get_device()

# Model and dataset
model_name = "bert-base-uncased"
max_length = 256
batch_size = 16 
gradient_accumulation_steps = 4 

# Training hyperparameters
num_epochs = 3
learning_rate = 2e-5
warmup_steps = 500
weight_decay = 0.01

# rSVDAdam specific parameters
rank_fraction = 0.3 
proj_interval = 500  # Recompute projector every 500 steps
use_rgp = True

# Evaluation
eval_batch_size = 32
save_dir = "./bert_sst2_checkpoints"
os.makedirs(save_dir, exist_ok=True)

print(f"Using device: {device}")
print(f"Model: {model_name}")
print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")


# ============================================================
# Load Dataset
# ============================================================
print("\nLoading SST-2 dataset...")
dataset = load_dataset("glue", "sst2")


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    """Tokenize the examples."""
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

print("Tokenizing dataset...")
tokenized_train = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=["sentence", "idx"]
)
tokenized_val = dataset["validation"].map(
    tokenize_function,
    batched=True,
    remove_columns=["sentence", "idx"]
)

print(f"Train samples: {len(tokenized_train)}")
print(f"Validation samples: {len(tokenized_val)}")

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
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])


# ============================================================
# Trainer Setup
# ============================================================

def compute_metrics(eval_pred):
    """Compute accuracy from logits and labels."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}


class RsvdTrainer(Trainer):
    """Custom Trainer that uses rSVDAdam and a linear warmup scheduler."""

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
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )


training_args = TrainingArguments(
    output_dir=save_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    warmup_steps=warmup_steps,
    eval_strategy="epoch",  
    save_strategy="epoch",  
    logging_dir=os.path.join(save_dir, "logs"),
    logging_steps=50,
    save_total_limit=3,
)


print("\n" + "=" * 70)
print("Starting Training with Hugging Face Trainer")
print("=" * 70)

# Optional memory tracking before training
reset_memory_stats(device)
initial_memory = get_current_memory_mb(device)
if initial_memory is not None:
    print(f"Initial GPU memory: {initial_memory:.2f} MB")


trainer = RsvdTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

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

print(f"\nModel checkpoints and logs saved to: {save_dir}")

