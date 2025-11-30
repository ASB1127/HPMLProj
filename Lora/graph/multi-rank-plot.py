import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================================
# CONFIG: all ranks to load
# =========================================
RANKS = [4, 8, 16]
BASE = "."          # <--- FIXED (your folders are in current directory)

os.makedirs("./plots", exist_ok=True)

# Storage for comparison plots
memory_summary = {}
flops_summary = {}
program_mem_summary = {}

# =========================================
# LOAD PER-RANK DATA
# =========================================
data = {}

for r in RANKS:
    rank_dir = f"{BASE}/r{r}"

    data[r] = {
        "epoch_mem": pd.read_csv(f"{rank_dir}/epoch_peak_memory.csv"),
        "flops": pd.read_csv(f"{rank_dir}/flops_profiler_stats.csv"),
        "total_mem": pd.read_csv(f"{rank_dir}/total_program_memory.csv"),
        "loss": pd.read_csv(f"{rank_dir}/epoch_loss.csv"),
    }

    # Summaries for comparison plots
    memory_summary[r] = data[r]["epoch_mem"]["peak_memory_bytes"].max() / 1e9
    flops_summary[r] = data[r]["flops"].loc[data[r]["flops"]["metric"] == "step_flops", "value"].item()
    program_mem_summary[r] = (
        data[r]["total_mem"]
        .loc[data[r]["total_mem"]["metric"] == "program_total_peak_memory", "value_bytes"]
        .item() / 1e9
    )

# =========================================
# PLOT 1 — Loss curves for all ranks
# =========================================
plt.figure(figsize=(8,6))
for r in RANKS:
    df = data[r]["loss"]
    plt.plot(df["epoch"], df["train_loss"], marker="o", label=f"Rank {r} — Train Loss")
    if df["eval_loss"].notna().any():
        plt.plot(df["epoch"], df["eval_loss"], marker="o", linestyle="--", label=f"Rank {r} — Eval Loss")

plt.title("Loss per Epoch Across LoRA Ranks", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/loss_across_ranks.png", dpi=300)
plt.close()

# =========================================
# PLOT 2 — Peak GPU Memory per Epoch (all ranks)
# =========================================
plt.figure(figsize=(8,6))
for r in RANKS:
    df = data[r]["epoch_mem"]
    plt.plot(df["epoch"], df["peak_memory_bytes"] / 1e9, marker="o", label=f"Rank {r}")

plt.title("Peak GPU Memory per Epoch Across Ranks", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Peak Memory (GB)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/peak_memory_per_epoch_across_ranks.png", dpi=300)
plt.close()

# =========================================
# PLOT 3 — FLOPs vs Rank
# =========================================
plt.figure(figsize=(6,5))
plt.bar([f"r{r}" for r in RANKS], [flops_summary[r] / 1e12 for r in RANKS], 
        color=["#1f77b4", "#ff7f0e", "#2ca02c"])
plt.ylabel("FLOPs per Step (TFLOPs)")
plt.title("Compute Cost vs LoRA Rank")
plt.tight_layout()
plt.savefig("./plots/flops_vs_rank.png", dpi=300)
plt.close()

# =========================================
# PLOT 4 — Peak GPU Memory vs Rank
# =========================================
plt.figure(figsize=(6,5))
plt.bar([f"r{r}" for r in RANKS], [memory_summary[r] for r in RANKS], 
        color=["#9467bd", "#d62728", "#8c564b"])
plt.ylabel("Peak Memory (GB)")
plt.title("Peak GPU Memory vs LoRA Rank")
plt.tight_layout()
plt.savefig("./plots/peak_memory_vs_rank.png", dpi=300)
plt.close()

# =========================================
# PLOT 5 — Total Program Memory vs Rank
# =========================================
plt.figure(figsize=(6,5))
plt.bar([f"r{r}" for r in RANKS], [program_mem_summary[r] for r in RANKS], 
        color=["#bcbd22", "#17becf", "#7f7f7f"])
plt.ylabel("Memory (GB)")
plt.title("Total Program Peak Memory vs Rank")
plt.tight_layout()
plt.savefig("./plots/total_program_memory_vs_rank.png", dpi=300)
plt.close()

print("🔥 All multi-rank LoRA plots saved to: ./plots/")

