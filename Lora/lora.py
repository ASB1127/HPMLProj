from re import A
from transformers import TrainerCallback
import sys, os
import numpy as np
from torch.optim import AdamW
from torch.autograd import profiler as autograd_profiler

import evaluate
import csv



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



class MemoryPeakPerEpochCallback(TrainerCallback):
    def __init__(self, csv_path="epoch_peak_memory.csv"):
        self.csv_path = csv_path

        
        with open(self.csv_path, "w") as f:
            f.write("epoch,peak_memory_bytes\n")

    def on_epoch_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()


            print(f"[Epoch {int(state.epoch)}] Peak CUDA Memory: {peak/1e6:.2f} MB")

 
            with open(self.csv_path, "a") as f:
                f.write(f"{int(state.epoch)},{peak}\n")

class lora_run():
    
    def __init__(self, num_train_epochs, rank, learning_rate):
        self.num_train_epochs = num_train_epochs
        self.rank = rank
        self.learning_rate = learning_rate
        self.accuracy_metric = None
        self.tokenizer = None
        
    
    def compute_metrics(self,eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return self.accuracy_metric.compute(predictions=preds, references=labels)
        
    def tokenize_fn(self,examples):
            return self.tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
        
    def run(self):
    
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        memory_peak_callback = MemoryPeakPerEpochCallback()

        self.accuracy_metric = evaluate.load("accuracy")


        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        dataset = load_dataset("glue", "sst2")
        model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

     
        tokenized_ds = dataset.map(self.tokenize_fn, batched=True)
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])




        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        lora_config = LoraConfig(
            r=self.rank,                          
            lora_alpha=16,                
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["q_lin","k_lin","v_lin"],
        
        )

        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        model = model.to(device)

        args = TrainingArguments(
            output_dir="./distilbert-sst2-lora",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=50,
            report_to="none",
            fp16 = False,
            bf16 = False,
        )
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        scheduler = LambdaLR(optimizer, lambda _: 1.0)


            
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            optimizers=(optimizer, scheduler),
            callbacks=[memory_peak_callback],   
            
        )

        train_dataloader = trainer.get_train_dataloader()

        train_iter = iter(train_dataloader)
        for _ in range(3):
            batch = next(iter(train_dataloader))
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optimizer.step()

        if device == "cuda":

            batch = next(iter(train_dataloader))
            batch = {k: v.to(device) for k, v in batch.items()}

            with autograd_profiler.profile(
                with_flops=True,
                use_cuda=True,
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        
            total_flops_step = sum(
                e.flops for e in prof.key_averages() if e.flops is not None
            )
            print(f"[Profiler] FLOPs for one training step: {total_flops_step:,}")


            num_batches = len(train_dataloader)
            flops_per_epoch = total_flops_step * num_batches
            print(f"[Profiler] FLOPs per epoch (approx): {flops_per_epoch:,}")


            with open("flops_profiler_stats.csv", "w") as f:
                f.write("metric,value\n")
                f.write(f"step_flops,{total_flops_step}\n")
                f.write(f"epoch_flops,{flops_per_epoch}\n")

            


        trainer.train()
            
        trainer.evaluate()



        if torch.cuda.is_available():
            total_peak = torch.cuda.max_memory_reserved()
            print(f"[PROGRAM TOTAL PEAK GPU MEMORY]: {total_peak/1e6:.2f} MB")
            with open("total_program_memory.csv", "w") as f:
                f.write("metric,value_bytes\n")
                f.write(f"program_total_peak_memory,{total_peak}\n")

        model.save_pretrained("distilbert-sst2-lora")

        model = model.merge_and_unload()
        model.save_pretrained("distilbert-sst2-full")

