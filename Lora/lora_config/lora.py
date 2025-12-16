from re import A
from transformers import TrainerCallback
import sys, os
import numpy as np
from torch.optim import AdamW
from torch.autograd import profiler as autograd_profiler
from pathlib import Path
from peft import PeftModel

import evaluate
import csv
import shutil
        


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from datasets import load_dataset
from torch.profiler import profile, ProfilerActivity
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification



import torch



class MemoryPeakPerEpochCallback(TrainerCallback):
    def __init__(self, rank, dataset_name, base_path="./graph"):
        self.rank = rank
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.path = f"{base_path}/{dataset_name}/r{rank}"

        os.makedirs(self.path, exist_ok=True)

        self.csv_path = f"{self.path}/epoch_peak_memory.csv"
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

class LossPerEpochCallback(TrainerCallback):
    def __init__(self, rank, dataset_name, base_path="./graph"):
        self.rank = rank
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.path = f"{base_path}/{dataset_name}/r{rank}"
        os.makedirs(self.path, exist_ok=True)

        self.csv_path = f"{self.path}/epoch_loss.csv"
        with open(self.csv_path, "w") as f:
            f.write("epoch,train_loss,eval_loss\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        # TRAIN loss is available inside `state.log_history`
        train_loss = None
        eval_loss  = None

        # Parse log history backwards until we find the last loss entries
        for log in reversed(state.log_history):
            if train_loss is None and "loss" in log:
                train_loss = log["loss"]
            if eval_loss is None and "eval_loss" in log:
                eval_loss = log["eval_loss"]
            if train_loss is not None and eval_loss is not None:
                break

        # If still missing, fill with blank or 0
        train_loss = train_loss if train_loss is not None else ""
        eval_loss = eval_loss if eval_loss is not None else ""

        # Append to CSV
        with open(self.csv_path, "a") as f:
            f.write(f"{int(state.epoch)},{train_loss},{eval_loss}\n")

        print(f"[Epoch {int(state.epoch)}] train_loss={train_loss}, eval_loss={eval_loss}")


class lora_run():
    
    def __init__(self, num_train_epochs, rank, learning_rate, dataset_name):    
        self.num_train_epochs = num_train_epochs
        self.rank = rank
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.accuracy_metric = None
        self.tokenizer = None
        
    
    def compute_metrics(self,eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return self.accuracy_metric.compute(predictions=preds, references=labels)

    def tokenize_fn(self, examples):
        if self.dataset_name == "sst2":
            text_field = "sentence"
        elif self.dataset_name == "imdb":
            text_field = "text"
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        return self.tokenizer(examples[text_field],truncation=True,padding="max_length",max_length=128)   
    
        
    def run(self):
    
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        memory_peak_callback = MemoryPeakPerEpochCallback(rank=self.rank, dataset_name=self.dataset_name)
        loss_callback = LossPerEpochCallback(rank=self.rank,dataset_name=self.dataset_name)
        rank_dir = f"./graph/{self.dataset_name}/r{self.rank}"
        os.makedirs(rank_dir, exist_ok=True)

        self.accuracy_metric = evaluate.load("accuracy")


        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        if self.dataset_name == "sst2":
            dataset = load_dataset("glue", "sst2")
        elif self.dataset_name == "imdb":
            dataset = load_dataset("imdb")
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

     
        tokenized_ds = dataset.map(self.tokenize_fn, batched=True)
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])




        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        lora_config = LoraConfig(
            r=self.rank,                          
            lora_alpha=self.rank,                
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["q_lin","k_lin","v_lin"],
        
        )

        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        model = model.to(device)

        args = TrainingArguments(
            output_dir=f"./distilbert-{self.dataset_name}-lora",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=50,
            report_to="none",
            fp16 = False,
            bf16 = False,
        )
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        scheduler = LambdaLR(optimizer, lambda _: 1.0)

        if self.dataset_name == "sst2":
            eval_split = "validation"
        elif self.dataset_name == "imdb":
            eval_split = "test"
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds[eval_split],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            optimizers=(optimizer, scheduler),
            callbacks=[memory_peak_callback, loss_callback],   
            
        )

        train_dataloader = trainer.get_train_dataloader()

        train_iter = iter(train_dataloader)
        for _ in range(3):
            batch =next(train_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optimizer.step()

        if device == "cuda":

            batch = next(train_iter)
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


            with open(f"{rank_dir}/flops_profiler_stats.csv", "w") as f:
                f.write("metric,value\n")
                f.write(f"step_flops,{total_flops_step}\n")
                f.write(f"epoch_flops,{flops_per_epoch}\n")

            


        trainer.train()
            
        trainer.evaluate()



        if torch.cuda.is_available():
            total_peak = torch.cuda.max_memory_reserved()
            print(f"[PROGRAM TOTAL PEAK GPU MEMORY]: {total_peak/1e6:.2f} MB")
            with open(f"{rank_dir}/total_program_memory.csv", "w") as f:
                f.write("metric,value_bytes\n")
                f.write(f"program_total_peak_memory,{total_peak}\n")
            
        adapter_dir = f"distilbert-{self.dataset_name}-lora-r{self.rank}"
        model.save_pretrained(adapter_dir)
        print(f"[Rank {self.rank}] Saved LoRA adapter to {adapter_dir}")

        # 2. Load ORIGINAL DistilBERT base model (stored locally)
        BASE = "distilbert-base-uncased"
        base_model_full = AutoModelForSequenceClassification.from_pretrained(BASE)

        # 3. Load the LoRA adapter we just saved
        model_with_adapter = PeftModel.from_pretrained(base_model_full, adapter_dir)

        # 4. Merge LoRA weights into a full model
        merged = model_with_adapter.merge_and_unload()

        # 5. Save merged full model to rank-specific folder
        full_model_dir = f"distilbert-{self.dataset_name}-full-r{self.rank}"
        merged.save_pretrained(full_model_dir)
        print(f"[Rank {self.rank}] Saved merged full model to {full_model_dir}")
        self.tokenizer.save_pretrained(full_model_dir)
        # 6. Copy tokenizer files into the merged-model directory
        print(f"[Rank {self.rank}] Tokenizer files copied into {full_model_dir}")

