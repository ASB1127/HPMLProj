from re import A
import sys, os
import numpy as np
import evaluate
import gc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from datasets import load_dataset
from torch.profiler import profile, ProfilerActivity
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from rsvd_svt import rSVDSVTAdam
import torch

class ProfilerCallback(TrainerCallback):
    def __init__(self, profile_steps=100):
        self.profiler = None
        self.profile_steps = profile_steps
        self.epoch_num = 0
        self.step_in_epoch = 0
        self.profiling_active = False
        
    def on_epoch_begin(self, args, state, control, **kwargs): 
        self.step_in_epoch = 0
        self.profiling_active = False

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
            labels = batch["labels"].to(device)
        
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

                with open(f"./profile_cuda_time_epoch_{epoch_label}.txt", "w") as f:
                    f.write(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=200))
                
                with open(f"./profile_cuda_memory_epoch_{epoch_label}.txt", "w") as f:
                    f.write(self.profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=200))
                
                print(f"\n=== Epoch {epoch_label} Profile Saved ===")
                print("\n=== TOP 10 TIME (CUDA) ===")
                print(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

                print("\n=== TOP 10 GPU MEMORY ===")
                print(self.profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

                trace_file = f"./trace_epoch_{epoch_label}.json"
                self.profiler.export_chrome_trace(trace_file)

                self.profiler = None
                torch.cuda.empty_cache()
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.profiling_active and self.profiler:
            self.profiler.__exit__(None, None, None)
            self.profiling_active = False

            self.profiler = None
            torch.cuda.empty_cache()
        
        self.epoch_num += 1
            


accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

dataset = load_dataset("imdb")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
tokenized_ds = dataset.map(tokenize_fn, batched=True)

tokenized_ds = tokenized_ds.rename_column("label", "labels")
tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

args = TrainingArguments(
    output_dir="./distilbert-imdb-rsvd-svt",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    learning_rate=2e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
    report_to="none",
    fp16 = False,
    bf16 = False,
)

optimizer = rSVDSVTAdam(
    model.parameters(),
    lr=2e-4,
    rank_fraction=0.3, # Tune this
    proj_interval=500, # Per Atith
    use_rgp=True, # Per Atith
    weight_decay=0.01, # Per Atith
    decoupled_weight_decay=True,  
    verbose_memory_once=True
)

scheduler = LambdaLR(optimizer, lambda _: 1.0)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
    data_collator=data_collator,
    callbacks=[ProfilerCallback(profile_steps=50)]
)

train_dataloader = trainer.get_train_dataloader()

for _ in range(2):
    batch = next(iter(train_dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad()
    out = model(**batch)
    out.loss.backward()
    optimizer.step()
    del batch, out
    torch.cuda.empty_cache()
    gc.collect()

torch.cuda.synchronize()

trainer.train()
trainer.evaluate()