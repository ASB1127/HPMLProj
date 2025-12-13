from re import A
from transformers import TrainerCallback
import sys, os
import numpy as np
from torch.optim import AdamW
from torch.autograd import profiler as autograd_profiler
from pathlib import Path, PurePath
from peft import PeftModel
from torch.autograd import profiler as autograd_profiler
from custom_optim.topr_adamw import TopRAdamW

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
    def __init__(self, rank, dataset_name, top_r, base_path="./graph"):
        self.rank = rank
        self.base_path = base_path
        self.top_r = top_r
        self.path = f"{base_path}/{dataset_name}/r{rank}/topr{top_r}"
        

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
    def __init__(self, rank, dataset_name, top_r, base_path="./graph"):
        self.rank = rank
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.top_r = top_r
        self.path = f"{base_path}/{dataset_name}/r{rank}/topr{top_r}"
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
    
    def __init__(self, num_train_epochs, rank, learning_rate, dataset_name, top_r: float = 0.1):
        self.num_train_epochs = num_train_epochs
        self.rank = rank
        self.learning_rate = learning_rate
        self.dataset_name = dataset_name
        assert 0 < top_r <= 1.0, "top_r must be in (0, 1]."
        self.top_r = float(top_r)
        self.accuracy_metric = None
        self.tokenizer = None
        
    
    def compute_metrics(self,eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return self.accuracy_metric.compute(predictions=preds, references=labels)
        
    def tokenize_fn(self,examples):
        if self.dataset_name == "sst2":
            text_field = "sentence"
        elif self.dataset_name == "imdb":
            text_field = "text"
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        return self.tokenizer(examples[text_field],truncation=True,padding="max_length",max_length=128)


    def compute_keep_fraction(self, model):
        total, nz = 0, 0
        for name, p in model.named_parameters():
            if "lora_" not in name or p.grad is None:
                continue
            g = p.grad
            total += g.numel()
            nz += (g != 0).sum().item()
        return nz / total if total > 0 else 1.0

    def warn_if_topr_too_small(self, model):
        if self.top_r >= 1.0:
            return
        total_tensors = 0
        zeroed_tensors = 0
        for name, p in model.named_parameters():
            if "lora_" not in name:
                continue
            numel = p.numel()
            if numel == 0:
                continue
            total_tensors += 1
            k = int(self.top_r * numel)
            if k <= 0:
                zeroed_tensors += 1

        if total_tensors > 0 and zeroed_tensors > 0:
            frac = zeroed_tensors / total_tensors
            print(
                f"[Top-R WARNING] top_r={self.top_r} causes k=0 for {zeroed_tensors}/{total_tensors} "
                f"LoRA tensors ({frac:.1%}); their grads will be fully masked."
            )
        
    def run(self):
    
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        memory_peak_callback = MemoryPeakPerEpochCallback(rank=self.rank,dataset_name=self.dataset_name,top_r=self.top_r)
        loss_callback = LossPerEpochCallback(rank=self.rank,dataset_name=self.dataset_name,top_r=self.top_r)
        rank_dir = f"./graph/{self.dataset_name}/r{self.rank}/topr{self.top_r}"
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
        self.warn_if_topr_too_small(model)

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
        optimizer = TopRAdamW(model.parameters(), lr=self.learning_rate, top_r=self.top_r)
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
        if device == "cuda":
            train_iter = iter(train_dataloader)
            for _ in range(3):
                batch = next(train_iter)
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                out = model(**batch)
                out.loss.backward()
                optimizer.step()
                
            batch = next(train_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            with autograd_profiler.profile(with_flops=True, use_cuda=True) as prof:
                out = model(**batch)
                loss = out.loss
                loss.backward()
                optimizer.step()
                
            masked_flops = sum(e.flops for e in prof.key_averages() if e.flops)
            num_batches = len(train_dataloader)
            masked_epoch_flops = masked_flops * num_batches
            
            print(f"[Profiler] Masked FLOPs per step: {masked_flops:,}")
            print(f"[Profiler] Masked FLOPs per epoch: {masked_epoch_flops:,}")
            keep_fraction = self.compute_keep_fraction(model)
            print(f"[Profiler] Top-r keep fraction: {keep_fraction:.4f}")
            
            effective_flops = masked_flops * keep_fraction
            effective_epoch_flops = effective_flops * num_batches
            
            print(f"[Profiler] Effective FLOPs per step: {effective_flops:,.0f}")
            print(f"[Profiler] Effective FLOPs per epoch: {effective_epoch_flops:,.0f}")
            
            with open(f"{rank_dir}/flops_profiler_stats.csv", "w") as f:
                f.write("metric,value\n")
                f.write(f"masked_step_flops,{masked_flops}\n")
                f.write(f"masked_epoch_flops,{masked_epoch_flops}\n")
                f.write(f"topr_keep_fraction,{keep_fraction}\n")
                f.write(f"topr_effective_step_flops,{effective_flops}\n")
                f.write(f"topr_effective_epoch_flops,{effective_epoch_flops}\n")

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
        # 6. Copy tokenizer files into the merged-model directory
        print(f"[Rank {self.rank}] Tokenizer files copied into {full_model_dir}")
