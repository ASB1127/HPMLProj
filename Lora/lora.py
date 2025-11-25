from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification


import torch

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

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("distilbert-sst2-lora")

model = model.merge_and_unload()
model.save_pretrained("distilbert-sst2-full")
