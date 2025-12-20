"""
Randomized SVD (rSVD) fine-tuning configuration and runner.
This module provides classes for fine-tuning models using the rSVDAdam optimizer,
tracking memory usage, loss, and accuracy on training subsets.
"""
import sys
import os
import numpy as np
import torch
from torch.autograd import profiler as autograd_profiler
from torch.utils.data import DataLoader, Subset
from transformers import TrainerCallback, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from optimizer.rSVD_adam_optimizer import rSVDAdam
from utils import get_device, count_parameters

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def _get_rank_dir(rank, base_path="./graph"):
    """Helper function to create consistent directory name from rank."""
    return f"{base_path}/r{rank}"


def _rank_to_rank_fraction(rank, typical_dim=768):
    """
    Convert rank (integer) to rank_fraction for optimizer.
    
    For DistilBERT, typical attention dimensions are ~768, so we estimate
    rank_fraction = rank / typical_dim. The optimizer will then calculate
    the actual rank as min(rank_fraction * min(m,n), rank) per parameter.
    
    Args:
        rank: Target rank (integer)
        typical_dim: Typical dimension for the model (default 768 for DistilBERT)
    
    Returns:
        rank_fraction: Fraction to pass to optimizer
    """
    return rank / typical_dim


class MemoryPeakPerEpochCallback(TrainerCallback):
    """
    Trainer callback to track and log peak CUDA memory usage at the start and end of each epoch.
    Stats are saved to a CSV file in the rank-specific results directory.
    """
    def __init__(self, rank, base_path="./graph"):
        self.rank = rank
        self.base_path = base_path
        # Create directory name based on rank
        self.path = _get_rank_dir(rank, base_path)
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
    """
    Trainer callback to log training and evaluation loss at the end of each epoch.
    Stats are saved to a CSV file in the rank-specific results directory.
    """
    def __init__(self, rank, base_path="./graph"):
        self.rank = rank
        self.base_path = base_path
        self.path = _get_rank_dir(rank, base_path)
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


class AccuracyOnTrainSubsetCallback(TrainerCallback):
    """
    Callback to track accuracy on a random subset of training data.
    
    This samples a fixed subset of the training data (with a fixed seed) and
    evaluates the model on this subset at the end of each epoch.
    """
    def __init__(self, rank, train_dataset, subset_size=1000, random_seed=42, 
                 base_path="./graph", device=None):
        """
        Args:
            rank: Rank identifier for directory naming
            train_dataset: The training dataset to sample from
            subset_size: Number of samples to include in the subset
            random_seed: Random seed for reproducible sampling
            base_path: Base path for saving results
            device: Device to use for evaluation (if None, will auto-detect)
        """
        self.rank = rank
        self.base_path = base_path
        self.path = _get_rank_dir(rank, base_path)
        os.makedirs(self.path, exist_ok=True)
        self.csv_path = f"{self.path}/epoch_train_accuracy.csv"
        
        # Sample subset with fixed seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        dataset_size = len(train_dataset)
        subset_size = min(subset_size, dataset_size)
        
        # Randomly sample indices
        indices = np.random.choice(dataset_size, size=subset_size, replace=False)
        self.subset_indices = sorted(indices.tolist())  # Sort for reproducibility
        
        # Create subset dataset
        self.subset_dataset = Subset(train_dataset, self.subset_indices)
        
        # Set up device
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        # Initialize CSV file
        with open(self.csv_path, "w") as f:
            f.write("epoch,train_accuracy\n")
        
        print(f"[Rank {rank}] Created training subset with {subset_size} samples (seed={random_seed})")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Evaluate model on training subset at end of each epoch."""
        model = kwargs.get('model')
        if model is None:
            return
        
        model.eval()
        correct = 0
        total = 0
        
        # Create DataLoader for subset
        dataloader = DataLoader(
            self.subset_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Compute accuracy
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Save to CSV
        with open(self.csv_path, "a") as f:
            f.write(f"{int(state.epoch)},{accuracy}\n")
        
        print(f"[Rank {self.rank}] Epoch {int(state.epoch)} - Train Subset Accuracy: {accuracy*100:.2f}%")
        
        model.train()  # Set back to training mode


class RsvdTrainer(Trainer):
    """
    Custom Hugging Face Trainer that initializes the rSVDAdam optimizer.
    """
    """Custom Trainer that uses rSVDAdam."""
    
    def __init__(self, rank, proj_interval, use_rgp, typical_dim=768, *args, **kwargs):
        self.rank = rank
        self.rank_fraction = _rank_to_rank_fraction(rank, typical_dim)
        self.proj_interval = proj_interval
        self.use_rgp = use_rgp
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Initializes the rSVDAdam optimizer and a dummy scheduler."""
        if self.optimizer is None:
            self.optimizer = rSVDAdam(
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


class rsvd_run():
    """
    Main runner class for rSVD fine-tuning experiments.
    Handles dataset loading, model initialization, training with rSVDAdam,
    and performance profiling.
    """
    
    def __init__(self, num_train_epochs, rank, learning_rate, dataset="sst", 
                 proj_interval=500, use_rgp=True, gradient_accumulation_steps=4,
                 base_path="./graph", typical_dim=768, train_subset_size=1000, 
                 train_subset_seed=42):
        self.num_train_epochs = num_train_epochs
        self.rank = rank
        self.rank_fraction = _rank_to_rank_fraction(rank, typical_dim)
        self.learning_rate = learning_rate
        self.dataset = dataset.lower()
        self.proj_interval = proj_interval
        self.use_rgp = use_rgp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.base_path = base_path
        self.train_subset_size = train_subset_size
        self.train_subset_seed = train_subset_seed
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
                'save_dir': f'./bert_imdb_checkpoints_r{rank}',
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
                'save_dir': f'./bert_sst2_checkpoints_r{rank}',
                'model_name': 'distilbert-base-uncased',
                'max_length': 128,
                'batch_size': 32,
                'eval_batch_size': 64,
            }
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}. Choose 'imdb' or 'sst'")
    
    def compute_metrics(self, eval_pred):
        """Computes accuracy for model evaluation."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}
    
    def tokenize_fn(self, examples):
        """Tokenizes the input text based on the selected dataset configuration."""
        return self.tokenizer(
            examples[self.dataset_config['text_column']],
            truncation=True,
            padding="max_length",
            max_length=self.dataset_config['max_length']
        )
    
    def run(self):
        """
        Executes the full rSVD fine-tuning workflow:
        1. Setup device and load dataset.
        2. Tokenize data and initialize callbacks.
        3. Load base model and transition to device.
        4. Configure and run the RsvdTrainer.
        5. Profile FLOPs (if CUDA is available).
        6. Save the fine-tuned model.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.device = get_device()
        
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

        memory_peak_callback = MemoryPeakPerEpochCallback(
            rank=self.rank,
            base_path=self.base_path
        )
        loss_callback = LossPerEpochCallback(
            rank=self.rank,
            base_path=self.base_path
        )
        
        # Create accuracy callback for training subset (after dataset is loaded)
        train_subset_callback = AccuracyOnTrainSubsetCallback(
            rank=self.rank,
            train_dataset=tokenized_ds[self.dataset_config['train_split']],
            subset_size=self.train_subset_size,
            random_seed=self.train_subset_seed,
            base_path=self.base_path,
            device=self.device
        )
        
        rank_dir = _get_rank_dir(self.rank, self.base_path)

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
        param_counts = count_parameters(self.model)
        print(f"Total parameters: {param_counts['total']:,}")
        print(f"Trainable parameters: {param_counts['trainable']:,}")

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

        trainer = RsvdTrainer(
            rank=self.rank,
            proj_interval=self.proj_interval,
            use_rgp=self.use_rgp,
            model=self.model,
            args=args,
            train_dataset=tokenized_ds[self.dataset_config['train_split']],
            eval_dataset=tokenized_ds[self.dataset_config['eval_split']],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[memory_peak_callback, loss_callback, train_subset_callback],
        )

        train_dataloader = trainer.get_train_dataloader()
        
        # Warmup steps
        optimizer = rSVDAdam(
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
            batch = next(iter(train_dataloader))
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            optimizer.zero_grad()
            out = self.model(**batch)
            out.loss.backward()
            optimizer.step()

        if self.device == "cuda":
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

        if torch.cuda.is_available():
            total_peak = torch.cuda.max_memory_reserved()
            print(f"[PROGRAM TOTAL PEAK GPU MEMORY]: {total_peak/1e6:.2f} MB")
            with open(f"{rank_dir}/total_program_memory.csv", "w") as f:
                f.write("metric,value_bytes\n")
                f.write(f"program_total_peak_memory,{total_peak}\n")
        
        print(f"[Rank {self.rank}] Saved model to {self.dataset_config['save_dir']}")

