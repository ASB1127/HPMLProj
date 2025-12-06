import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from torch.autograd import profiler as autograd_profiler


class profiler_forward():
    def __init__(self, rank):
        self.rank = rank
        self.MODEL_DIR = f"/workspace/HPMLProj/rSVD_Weight_Reduction/distilbert-sst2-rSVD-r{rank}"

        self.DEVICE = "cuda"

    def load_model(self):
        print(f"[Rank {self.rank}] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_DIR,
            local_files_only=True
        )

        print(f"[Rank {self.rank}] Loading merged full model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_DIR,
            local_files_only=True
        ).to(self.DEVICE)

        model.eval()
        return tokenizer, model

    def profile_forward(self,tokenizer, model):
        text = "This movie was absolutely fantastic! Highly recommended."
        inputs = tokenizer(text, return_tensors="pt").to(self.DEVICE)
        SAVE_DIR = f"./graph/r{self.rank}/forward_pass"
        os.makedirs(SAVE_DIR, exist_ok=True)
        

        # ----------------------------------------------------
        # 1. PROFILE GPU MEMORY
        # ----------------------------------------------------
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(**inputs)

        peak_memory = torch.cuda.max_memory_allocated()
        with open(f"{SAVE_DIR}/peak_forward_memory.csv", "w") as f:
            f.write("metric,value_bytes\n")
            f.write(f"peak_memory,{peak_memory}\n")

        print(f"[Rank {self.rank}] Peak Forward Memory: {peak_memory/1e6:.2f} MB")

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
            
    def runner(self):
        tokenizer, model = self.load_model()
        self.profile_forward(tokenizer, model)

