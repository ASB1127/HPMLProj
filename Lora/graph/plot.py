"""
Plotting script for LoRA fine-tuning experiments.
Generates visualizations for training loss, peak memory usage, and FLOPs 
across different ranks and datasets. Includes a 2x2 summary plot.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

# ============================================================
# CONFIG
# ============================================================

DATASET = "sst2"
RANKS = [4, 8, 16, 64, 128]

ROOT = "."
PLOTS = f"{ROOT}/plots/{DATASET}"
os.makedirs(PLOTS, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def read_epoch_loss(rank):
    """Reads the epoch loss CSV for a given rank."""
    return pd.read_csv(f"{ROOT}/{DATASET}/r{rank}/epoch_loss.csv")

def read_peak_memory(rank):
    """Reads the peak memory CSV for a given rank and returns the maximum in MB."""
    df = pd.read_csv(f"{ROOT}/{DATASET}/r{rank}/epoch_peak_memory.csv")
    return df["peak_memory_bytes"].max() / 1e6  # MB

def read_step_flops(rank):
    """Reads the FLOPs profile stats for a given rank and returns the step FLOPs value."""
    df = pd.read_csv(f"{ROOT}/{DATASET}/r{rank}/flops_profiler_stats.csv")
    return float(df[df["metric"] == "step_flops"]["value"].iloc[0])

# ============================================================
# PLOT 1 — Training Loss vs Epoch (Across Ranks)
# ============================================================

def plot_train_loss_vs_epoch():
    """Plots training loss vs epoch for all configured ranks and saves it as an image."""
    plt.figure(figsize=(8, 5))

    for r in RANKS:
        df = read_epoch_loss(r)
        plt.plot(df["epoch"], df["train_loss"], marker="o", label=f"r={r}")

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("LoRA Training Loss vs Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{PLOTS}/train_loss_vs_epoch.png", dpi=300)
    plt.close()

# ============================================================
# PLOT 2 — Peak Memory vs Rank (Bar)
# ============================================================

def plot_peak_memory_vs_rank():
    """Plots peak GPU memory usage vs rank as a bar chart."""
    mems = [read_peak_memory(r) for r in RANKS]

    plt.figure(figsize=(7, 5))
    plt.bar([str(r) for r in RANKS], mems)

    plt.xlabel("Rank (r)")
    plt.ylabel("Peak GPU Memory (MB)")
    plt.title("LoRA Peak Memory vs Rank")
    plt.grid(axis="y")

    plt.tight_layout()
    plt.savefig(f"{PLOTS}/peak_memory_vs_rank.png", dpi=300)
    plt.close()

# ============================================================
# PLOT 3 — Step FLOPs vs Rank (Measured)
# ============================================================

def plot_step_flops_vs_rank():
    """Plots measured step FLOPs vs rank on a log scale."""
    flops = [read_step_flops(r) for r in RANKS]

    plt.figure(figsize=(7, 5))
    plt.plot(RANKS, flops, marker="o")
    plt.yscale("log")

    plt.xlabel("Rank (r)")
    plt.ylabel("Step FLOPs (log scale)")
    plt.title("LoRA Measured Step FLOPs vs Rank")
    plt.grid(True, which="both")

    plt.tight_layout()
    plt.savefig(f"{PLOTS}/step_flops_vs_rank.png", dpi=300)
    plt.close()

# ============================================================
# 2×2 SUMMARY — LoRA BASELINE
# ============================================================

def plot_lora_2x2():
    """
    Combines individual plots into a 2x2 summary mosaic showing:
    - Training Loss vs Epoch
    - Peak Memory vs Rank
    - Descriptive text for LoRA baseline
    - Step FLOPs vs Rank
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    img_loss  = plt.imread(f"{PLOTS}/train_loss_vs_epoch.png")
    img_mem   = plt.imread(f"{PLOTS}/peak_memory_vs_rank.png")
    img_flops = plt.imread(f"{PLOTS}/step_flops_vs_rank.png")

    # Top-left: Training loss vs epoch
    axes[0, 0].imshow(img_loss)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Training Loss vs Epoch")

    # Top-right: Peak memory vs rank
    axes[0, 1].imshow(img_mem)
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Peak Memory vs Rank")

    # Bottom-left: Leave blank or note
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.5, 0.5,
        "LoRA baseline:\nDense kernels\nRank-independent FLOPs",
        ha="center", va="center", fontsize=14
    )

    # Bottom-right: Step FLOPs vs rank
    axes[1, 1].imshow(img_flops)
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Measured Step FLOPs vs Rank")

    fig.suptitle(
        f"LoRA Baseline Summary — {DATASET.upper()}",
        fontsize=20,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{PLOTS}/lora_2x2_summary.png", dpi=300)
    plt.close()

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("Generating LoRA baseline plots...")

    plot_train_loss_vs_epoch()
    plot_peak_memory_vs_rank()
    plot_step_flops_vs_rank()
    plot_lora_2x2()

    print("✓ LoRA plots + 2×2 summary generated")

