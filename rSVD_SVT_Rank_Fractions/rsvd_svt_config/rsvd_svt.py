import sys
import os
import numpy as np
import torch
from torch.autograd import profiler as autograd_profiler
from transformers import TrainerCallback, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from optimizer.rSVD_SVT_adam_optimizer import rSVDSVTAdam

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def _get_rank_dir(rank_fraction, base_path="./graph"):
    """Helper function to create consistent directory name from rank_fraction."""
    return base_path + f"/r{rank_fraction}".replace(".", "_")

class MemoryPeakPerEpochCallback(TrainerCallback):
    def __init__(self, rank_fraction, base_path="./graph"):
        self.rank_fraction = rank_fraction
        self.base_path = base_path
        # Create directory name based on rank_fraction
        self.path = _get_rank_dir(rank_fraction, base_path)
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
    def __init__(self, rank_fraction, base_path="./graph"):
        self.rank_fraction = rank_fraction
        self.base_path = base_path
        self.path = _get_rank_dir(rank_fraction, base_path)
        os.makedirs(self.path, exist_ok=True)
        self.csv_path = f"{self.path}/epoch_loss.csv"
        with open(self.csv_path, "w") as f:
            f.write("epoch,train_loss,eval_loss\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        train_loss = None
        eval_loss = None
        for log in reversed(state.log_history):
            if train_loss is None and "loss" in log:
                train_loss = log["loss"]
            if eval_loss is None and "eval_loss" in log:
                eval_loss = log["eval_loss"]
            if train_loss is not None and eval_loss is not None:
                break
        train_loss = train_loss if train_loss is not None else ""
        eval_loss = eval_loss if eval_loss is not None else ""
        with open(self.csv_path, "a") as f:
            f.write(f"{int(state.epoch)},{train_loss},{eval_loss}\n")
        print(f"[Epoch {int(state.epoch)}] train_loss={train_loss}, eval_loss={eval_loss}")


class RsvdsvtTrainer(Trainer):
    """Custom Trainer that uses rSVDSVTAdam."""
    
    def __init__(self, rank_fraction, proj_interval, use_rgp, *args, **kwargs):
        self.rank_fraction = rank_fraction
        self.proj_interval = proj_interval
        self.use_rgp = use_rgp
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            self.optimizer = rSVDSVTAdam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                rank_fraction=self.rank_fraction,
                proj_interval=self.proj_interval,
                use_rgp=self.use_rgp,
                weight_decay=self.args.weight_decay,
                decoupled_weight_decay=True,
                verbose_memory_once=True,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = LambdaLR(self.optimizer, lambda _: 1.0)


class rsvd_svt_run():
    
    def __init__(self, num_train_epochs, rank_fraction, learning_rate, dataset="sst", 
                 proj_interval=500, use_rgp=True, gradient_accumulation_steps=4,
                 base_path="./graph"):
        self.num_train_epochs = num_train_epochs
        self.rank_fraction = rank_fraction
        self.learning_rate = learning_rate
        self.dataset = dataset.lower()
        self.proj_interval = proj_interval
        self.use_rgp = use_rgp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if self.dataset == "imdb":
            self.base_path = "./" + "imdbb" + "/" + base_path
        else:
            self.base_path = "./" + self.dataset + "/" + base_path
        self.device = None
        self.tokenizer = None
        self.model = None
        
        # Dataset configuration
        if self.dataset == 'imdb':
            self.dataset_config = {
                'name': 'imdb',
                'dataset_path': 'imdb',
                'text_column': 'text',
                'train_split': 'train',
                'eval_split': 'test',
                'remove_columns': ['text'],
                'save_dir': "./bert_imdb_checkpoints_" + f"r{rank_fraction}".replace(".", "_"),
                'model_name': 'distilbert-base-uncased',
                'max_length': 128,
                'batch_size': 32,
                'eval_batch_size': 64,
            }
        elif self.dataset == 'sst':
            self.dataset_config = {
                'name': 'sst2',
                'dataset_path': 'glue',
                'dataset_config': 'sst2',
                'text_column': 'sentence',
                'train_split': 'train',
                'eval_split': 'validation',
                'remove_columns': ['sentence', 'idx'],
                'save_dir': "./bert_sst_checkpoints_" + f"r{rank_fraction}".replace(".", "_"),
                'model_name': 'distilbert-base-uncased',
                'max_length': 128,
                'batch_size': 32,
                'eval_batch_size': 64,
            }
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}. Choose 'imdb' or 'sst'")
    
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}
    
    def tokenize_fn(self, examples):
        return self.tokenizer(
            examples[self.dataset_config['text_column']],
            truncation=True,
            padding="max_length",
            max_length=self.dataset_config['max_length']
        )
    
    def run(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        memory_peak_callback = MemoryPeakPerEpochCallback(
            rank_fraction=self.rank_fraction,
            base_path=self.base_path
        )
        loss_callback = LossPerEpochCallback(
            rank_fraction=self.rank_fraction,
            base_path=self.base_path
        )
        rank_dir = _get_rank_dir(self.rank_fraction, self.base_path)

        self.device = "cuda"
        
        # Load dataset
        print(f"\nLoading {self.dataset_config['name'].upper()} dataset...")
        if self.dataset == 'imdb':
            dataset = load_dataset(self.dataset_config['dataset_path'])
        else:  # sst
            dataset = load_dataset(
                self.dataset_config['dataset_path'], 
                self.dataset_config['dataset_config']
            )
        
        model_name = self.dataset_config['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        tokenized_ds = dataset.map(self.tokenize_fn, batched=True)
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # Load model
        print(f"\nLoading {model_name}...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        
        self.model = base_model.to(self.device)
        
        # Count parameters
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")

        args = TrainingArguments(
            output_dir=self.dataset_config['save_dir'],
            per_device_train_batch_size=self.dataset_config['batch_size'],
            per_device_eval_batch_size=self.dataset_config['eval_batch_size'],
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=50,
            report_to="none",
            fp16=False,
            bf16=False,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        trainer = RsvdsvtTrainer(
            rank_fraction=self.rank_fraction,
            proj_interval=self.proj_interval,
            use_rgp=self.use_rgp,
            model=self.model,
            args=args,
            train_dataset=tokenized_ds[self.dataset_config['train_split']],
            eval_dataset=tokenized_ds[self.dataset_config['eval_split']],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[memory_peak_callback, loss_callback],
        )

        train_dataloader = trainer.get_train_dataloader()
        
        # Warmup steps
        optimizer = rSVDSVTAdam(
            self.model.parameters(),
            lr=self.learning_rate,
            rank_fraction=self.rank_fraction,
            proj_interval=self.proj_interval,
            use_rgp=self.use_rgp,
            weight_decay=0.01,
            decoupled_weight_decay=True,
            verbose_memory_once=True,
        )
        
        train_iter = iter(train_dataloader)
        for _ in range(3):
            batch = next(train_iter)
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            optimizer.zero_grad()
            out = self.model(**batch)
            out.loss.backward()
            optimizer.step()

        batch = next(iter(train_dataloader))
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}

        with autograd_profiler.profile(
            with_flops=True,
            use_cuda=True,
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            optimizer.zero_grad()
            outputs = self.model(**batch)
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
        trainer.save_model(self.dataset_config['save_dir'])

        total_peak = torch.cuda.max_memory_reserved()
        print(f"[PROGRAM TOTAL PEAK GPU MEMORY]: {total_peak/1e6:.2f} MB")
        with open(f"{rank_dir}/total_program_memory.csv", "w") as f:
            f.write("metric,value_bytes\n")
            f.write(f"program_total_peak_memory,{total_peak}\n")
        
        print(f"[Rank Fraction {self.rank_fraction}] Saved model to {self.dataset_config['save_dir']}")

