from re import A
import sys, os
import numpy as np
from torch.optim import AdamW
import evaluate


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from datasets import load_dataset
from torch.profiler import profile, ProfilerActivity
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from custom_adam.optimizer.custom_adam_optimizer import CustomAdam



import torch

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

dataset = load_dataset("glue", "sst2")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
tokenized_ds = dataset.map(tokenize_fn, batched=True)
tokenized_ds = tokenized_ds.rename_column("label", "labels")
tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])




base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

lora_config = LoraConfig(
    r=8,                          
    lora_alpha=16,                
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_lin", "v_lin","query","value"],
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
model = model.to(device)

args = TrainingArguments(
    output_dir="./distilbert-sst2-lora",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    learning_rate=2e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
    report_to="none",
    fp16 = False,
    bf16 = False,
)
optimizer = CustomAdam(model.parameters(), lr=2e-4)
scheduler = LambdaLR(optimizer, lambda _: 1.0)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
)

train_dataloader = trainer.get_train_dataloader()

for _ in range(2):
    batch = next(iter(train_dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad()
    out = model(**batch)
    out.loss.backward()
    optimizer.step()

torch.cuda.synchronize()


batch = next(iter(train_dataloader))
batch = {k: v.to(device) for k, v in batch.items()}



optimizer.zero_grad()
outputs = model(**batch)
loss = outputs.loss
loss.backward()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    optimizer.step()

trainer.train()
trainer.evaluate()



print("\n=== TIME (CUDA) ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

print("\n=== GPU MEMORY ===")
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

with open("profile_cuda_time.txt", "w") as f:
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=200))

with open("profile_cuda_memory.txt", "w") as f:
    f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=200))

# Optional: Chrome trace
prof.export_chrome_trace("profile_trace.json")


model.save_pretrained("distilbert-sst2-lora")

model = model.merge_and_unload()
model.save_pretrained("distilbert-sst2-full")

