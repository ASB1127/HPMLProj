import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.autograd import profiler as autograd_profiler


def _get_rank_dir(rank_fraction, base_path="./graph"):
    """Helper function to create consistent directory name from rank_fraction."""
    # Convert rank_fraction to string and replace dots only in the number part
    rank_str = str(rank_fraction).replace(".", "_")
    return f"{base_path}/r{rank_str}"


class profiler_forward():
    def __init__(self, rank_fraction, dataset="sst", base_path="./graph"):
        self.rank_fraction = rank_fraction
        self.dataset = dataset.lower()
        self.base_path = base_path
        
        # Determine model directory based on dataset and rank_fraction
        if self.dataset == 'imdb':
            self.MODEL_DIR = f"./bert_imdb_checkpoints_r{rank_fraction}".replace(".", "_")
        else:  # sst
            self.MODEL_DIR = f"./bert_sst2_checkpoints_r{rank_fraction}".replace(".", "_")
        
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        print(f"[Rank Fraction {self.rank_fraction}] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_DIR,
            local_files_only=True
        )

        print(f"[Rank Fraction {self.rank_fraction}] Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_DIR,
            local_files_only=True
        ).to(self.DEVICE)

        model.eval()
        return tokenizer, model

    def profile_forward(self, tokenizer, model):
        # Sample text for forward pass
        if self.dataset == 'imdb':
            text = "This movie was absolutely fantastic! Highly recommended."
        else:  # sst
            text = "This movie was absolutely fantastic! Highly recommended."
        
        inputs = tokenizer(text, return_tensors="pt").to(self.DEVICE)
        rank_dir = _get_rank_dir(self.rank_fraction, self.base_path)
        SAVE_DIR = f"{rank_dir}/forward_pass"
        os.makedirs(SAVE_DIR, exist_ok=True)

        # ----------------------------------------------------
        # 1. PROFILE GPU MEMORY
        # ----------------------------------------------------
        if self.DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(**inputs)

            peak_memory = torch.cuda.max_memory_allocated()
            with open(f"{SAVE_DIR}/peak_forward_memory.csv", "w") as f:
                f.write("metric,value_bytes\n")
                f.write(f"peak_memory,{peak_memory}\n")

            print(f"[Rank Fraction {self.rank_fraction}] Peak Forward Memory: {peak_memory/1e6:.2f} MB")

            # ----------------------------------------------------
            # 2. PROFILE FLOPs FOR A SINGLE FORWARD PASS
            # ----------------------------------------------------
            with autograd_profiler.profile(
                with_flops=True,
                use_cuda=True,
                record_shapes=False,
                profile_memory=False,
            ) as prof:

                with torch.no_grad():
                    _ = model(**inputs)

            total_flops = sum(e.flops for e in prof.key_averages() if e.flops)
            print(f"[Forward] FLOPs: {total_flops:,}")

            with open(f"{SAVE_DIR}/forward_flops.csv", "w") as f:
                f.write("metric,value\n")
                f.write(f"forward_flops,{total_flops}\n")
        else:
            print(f"[Rank Fraction {self.rank_fraction}] Forward pass profiling skipped (CPU device)")
            
    def runner(self):
        tokenizer, model = self.load_model()
        self.profile_forward(tokenizer, model)

