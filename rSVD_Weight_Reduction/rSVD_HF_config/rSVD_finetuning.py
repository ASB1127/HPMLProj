from re import A
from transformers import TrainerCallback
import sys, os
import numpy as np
from torch.optim import AdamW
from torch.autograd import profiler as autograd_profiler
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub import HfFolder
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

from transformers import TrainingArguments, Trainer
from .modelcard.model_card import ModelCard


import torch

from .rSVD_modeling_distilbert import (
    rSVDLinear,
    apply_rsvd_to_attention_qkv,
    DistilBertForSequenceClassification_rSVD,
)



class MemoryPeakPerEpochCallback(TrainerCallback):
    def __init__(self, rank, base_path="./graph"):
        self.rank = rank
        self.path = f"{base_path}/r{rank}"
        os.makedirs(self.path, exist_ok=True)

        # CSV file for peak memory
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
    def __init__(self,rank, base_path="./graph"):
        self.rank = rank
        self.path = f"{base_path}/r{rank}"
        os.makedirs(self.path, exist_ok=True)

        # Create CSV file
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





class rSVD_run():
    
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
        total_flops_step = None
        flops_per_epoch = None
        total_peak = None
    
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        memory_peak_callback = MemoryPeakPerEpochCallback(rank=self.rank,base_path="./rSVD_finetuning")
        loss_callback = LossPerEpochCallback(rank=self.rank,base_path="./rSVD_finetuning")
        output_dir = "./rSVD_finetuning"
        os.makedirs(output_dir, exist_ok=True)

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



        base_model = DistilBertForSequenceClassification_rSVD.from_pretrained(
        model_name,
        num_labels=2,
        )

        base_model.config.is_rsvd_model = True
        base_model.config.rsvd_rank = self.rank
        base_model.config.architectures = ["DistilBertForSequenceClassification_rSVD"]


        model = base_model
        model = model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        apply_rsvd_to_attention_qkv(model, self.rank)
        for name, p in model.named_parameters():
            if ("A" in name) or ("B" in name) or ("C" in name) or ("classifier" in name):
                p.requires_grad = True

        model=model.to(device)
        args = TrainingArguments(
            output_dir="./distilbert-sst2-rSVD",
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
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.learning_rate)
        scheduler = LambdaLR(optimizer, lambda _: 1.0)


            
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            optimizers=(optimizer, scheduler),
            callbacks=[memory_peak_callback, loss_callback],   
            
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


            with open(f"{output_dir}/flops_profiler_stats.csv", "w") as f:
                f.write("metric,value\n")
                f.write(f"step_flops,{total_flops_step}\n")
                f.write(f"epoch_flops,{flops_per_epoch}\n")

            


        trainer.train()
            
        trainer.evaluate()



        if torch.cuda.is_available():
            total_peak = torch.cuda.max_memory_reserved()
            print(f"[PROGRAM TOTAL PEAK GPU MEMORY]: {total_peak/1e6:.2f} MB")
            with open(f"{output_dir}/total_program_memory.csv", "w") as f:
                f.write("metric,value_bytes\n")
                f.write(f"program_total_peak_memory,{total_peak}\n")

        full_model_dir = f"distilbert-sst2-rSVD-r{self.rank}"

        model.config.is_rsvd_model = True
        model.config.rsvd_rank = self.rank
        model.config.architectures = ["DistilBertForSequenceClassification_rSVD"]


        model.save_pretrained(full_model_dir)
        self.tokenizer.save_pretrained(full_model_dir)
        print(f"[Rank {self.rank}] Saved full dense model to {full_model_dir}")

        card = ModelCard()
        eval_results = trainer.evaluate()
        val_accuracy = eval_results.get("eval_accuracy", None)
        card.write_model_card(
        path=full_model_dir,
        rank=self.rank,
        flops_step=total_flops_step if device == "cuda" else None,
        flops_epoch=flops_per_epoch if device == "cuda" else None,
        peak_memory=total_peak if torch.cuda.is_available() else None,
        val_accuracy=val_accuracy,
        learning_rate=self.learning_rate,
        num_epochs=self.num_train_epochs,
        )

        TOKENIZER_SRC_DIR = "/workspace/HPMLProj/Lora/distilbert-original"
        for fname in ["vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            src = Path(TOKENIZER_SRC_DIR) / fname
            dst = Path(full_model_dir) / fname
            if src.exists():
                shutil.copy(src, dst)

        print(f"[Rank {self.rank}] Tokenizer files copied into {full_model_dir}")

        token = HfFolder.get_token()
        upload_enabled = token is not None

        if not upload_enabled:
            print("Not logged into HuggingFace Hub. Skipping upload.")
        else:
            print("Logged into HuggingFace Hub. Proceeding with upload...")

            repo_id = f"ab2720/distilbert-sst2-rSVD-r{self.rank}"

            api = HfApi()

            api.create_repo(
            repo_id=repo_id,
            exist_ok=True,
            )

            api.upload_folder(
            folder_path=full_model_dir,
            repo_id=repo_id,
            )
            print(f"[Rank {self.rank}] Uploaded model to the Hub at {repo_id}")

