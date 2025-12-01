import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default: look for data in parent directory's checkpoint folders
# Can override with command line argument
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "bert_imdb_checkpoints")

# Create plots directory
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"Loading data from: {DATA_DIR}")
print(f"Saving plots to: {PLOT_DIR}")

# ===============================
# Load CSV Files
# ===============================
epoch_loss = None
epoch_mem = None
epoch_gpu_time = None

if os.path.exists(os.path.join(DATA_DIR, "epoch_loss.csv")):
    epoch_loss = pd.read_csv(os.path.join(DATA_DIR, "epoch_loss.csv"))
    print("✓ Loaded epoch_loss.csv")
else:
    print("⚠ epoch_loss.csv not found")

if os.path.exists(os.path.join(DATA_DIR, "epoch_memory.csv")):
    epoch_mem = pd.read_csv(os.path.join(DATA_DIR, "epoch_memory.csv"))
    print("✓ Loaded epoch_memory.csv")
else:
    print("⚠ epoch_memory.csv not found")

if os.path.exists(os.path.join(DATA_DIR, "epoch_gpu_time.csv")):
    epoch_gpu_time = pd.read_csv(os.path.join(DATA_DIR, "epoch_gpu_time.csv"))
    print("✓ Loaded epoch_gpu_time.csv")
else:
    print("⚠ epoch_gpu_time.csv not found")

# ===============================
# PLOT 1 — Loss per Epoch
# ===============================
if epoch_loss is not None:
    plt.figure(figsize=(8, 5))
    
    if epoch_loss["train_loss"].notna().any():
        plt.plot(
            epoch_loss["epoch"],
            epoch_loss["train_loss"],
            marker="o",
            linewidth=2,
            label="Train Loss",
            color="#1f77b4"
        )
    
    if "eval_loss" in epoch_loss.columns and epoch_loss["eval_loss"].notna().any():
        plt.plot(
            epoch_loss["epoch"],
            epoch_loss["eval_loss"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label="Eval Loss",
            color="#ff7f0e"
        )
    
    plt.title("rSVDAdam Fine-Tuning: Loss per Epoch", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rsvd_loss_across_epochs.png"), dpi=300)
    plt.close()
    print("✓ Saved rsvd_loss_across_epochs.png")

# ===============================
# PLOT 2 — Peak GPU Memory per Epoch
# ===============================
if epoch_mem is not None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        epoch_mem["epoch"],
        epoch_mem["peak_allocated_bytes"] / 1e9,  # Convert to GB
        marker="o",
        linewidth=2,
        color="#9467bd"
    )
    plt.title("rSVDAdam Fine-Tuning: Peak GPU Memory per Epoch", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Peak Memory (GB)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rsvd_memory_across_epochs.png"), dpi=300)
    plt.close()
    print("✓ Saved rsvd_memory_across_epochs.png")

# ===============================
# PLOT 3 — GPU Time per Epoch
# ===============================
if epoch_gpu_time is not None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        epoch_gpu_time["epoch"],
        epoch_gpu_time["gpu_time_ms"] / 1000,  # Convert to seconds
        marker="o",
        linewidth=2,
        color="#2ca02c"
    )
    plt.title("rSVDAdam Fine-Tuning: GPU Time per Epoch", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("GPU Time (seconds)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rsvd_gpu_time_across_epochs.png"), dpi=300)
    plt.close()
    print("✓ Saved rsvd_gpu_time_across_epochs.png")

# ===============================
# PLOT 4 — Memory Breakdown
# ===============================
if epoch_mem is not None:
    peak_per_epoch_gb = epoch_mem["peak_allocated_bytes"].max() / 1e9
    avg_per_epoch_gb = epoch_mem["peak_allocated_bytes"].mean() / 1e9
    
    plt.figure(figsize=(8, 5))
    plt.bar(
        ["Peak per Epoch", "Average per Epoch"],
        [peak_per_epoch_gb, avg_per_epoch_gb],
        color=["#d62728", "#bcbd22"]
    )
    plt.title("GPU Memory Footprint of rSVDAdam Fine-Tuning", fontsize=14)
    plt.ylabel("Memory (GB)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rsvd_memory_breakdown.png"), dpi=300)
    plt.close()
    print("✓ Saved rsvd_memory_breakdown.png")

print(f"\n🔥 All rSVD plots saved to: {PLOT_DIR}/")

