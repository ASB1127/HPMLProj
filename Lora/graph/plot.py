import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

# ------------------------------
# CONFIG
# ------------------------------
DATASET = "imdb"   # or "sst2"
RANKS = [4, 8, 16, 64, 128]

os.makedirs(f"./plots/{DATASET}", exist_ok=True)


# ------------------------------
# HELPERS
# ------------------------------

def read_eval_loss(dataset, rank):
    df = pd.read_csv(f"./{dataset}/r{rank}/epoch_loss.csv")
    return df["eval_loss"].dropna().iloc[-1]

def read_peak_memory(dataset, rank):
    df = pd.read_csv(f"./{dataset}/r{rank}/epoch_peak_memory.csv")
    return df["peak_memory_bytes"].max() / 1e6  # MB

def read_step_flops(dataset, rank):
    df = pd.read_csv(f"./{dataset}/r{rank}/flops_profiler_stats.csv")
    return df[df["metric"] == "step_flops"]["value"].iloc[0]


# ============================================================
#   PLOT 1 — Eval Loss vs Epoch Across Ranks
# ============================================================

def plot_loss_across_ranks(dataset, ranks):
    plt.figure(figsize=(8,6))

    for r in ranks:
        df = pd.read_csv(f"./{dataset}/r{r}/epoch_loss.csv")
        plt.plot(df["epoch"], df["eval_loss"], marker="o", label=f"r={r}")

    plt.xlabel("Epoch")
    plt.ylabel("Eval Loss")
    plt.title(f"Eval Loss vs Epoch Across Ranks — {dataset.upper()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./plots/{dataset}/loss_across_ranks.png", dpi=300)
    plt.close()


# ============================================================
#   PLOT 2 — MEMORY vs RANK (BAR CHART)
# ============================================================

def plot_memory_vs_rank(dataset, ranks):
    mems = [read_peak_memory(dataset, r) for r in ranks]

    plt.figure(figsize=(7,5))
    plt.bar([str(r) for r in ranks], mems, color="purple")
    plt.xlabel("LoRA Rank (r)")
    plt.ylabel("Peak Memory (MB)")
    plt.title(f"Peak Memory vs LoRA Rank — {dataset.upper()}")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(f"./plots/{dataset}/memory_vs_rank.png", dpi=300)
    plt.close()


# ============================================================
#   PLOT 3 — Final Eval Loss vs Rank (CATEGORICAL AXIS)
# ============================================================

def plot_eval_loss_vs_rank(dataset, ranks):
    losses = [read_eval_loss(dataset, r) for r in ranks]

    plt.figure(figsize=(7,5))
    plt.plot(ranks, losses, marker="o")
    plt.xlabel("LoRA Rank (r)")
    plt.ylabel("Final Eval Loss")
    plt.title(f"Final Eval Loss vs LoRA Rank — {dataset.upper()}")
    plt.grid(True)
    plt.xticks(ranks, [str(r) for r in ranks])
    plt.tight_layout()
    plt.savefig(f"./plots/{dataset}/eval_loss_vs_rank.png", dpi=300)
    plt.close()


# ============================================================
#   PLOT 4 — FLOPs vs Rank (CATEGORICAL + LOG Y ONLY)
# ============================================================

def plot_flops_vs_rank(dataset, ranks):
    flops = [read_step_flops(dataset, r) for r in ranks]

    plt.figure(figsize=(7,5))
    plt.plot(ranks, flops, marker="o", color="steelblue")

    # Categorical x-axis
    plt.xticks(ranks, [str(r) for r in ranks])

    # Log-scale y-axis ONLY
    plt.yscale("log")

    plt.xlabel("LoRA Rank (r)")
    plt.ylabel("Step FLOPs (log scale)")
    plt.title(f"Step FLOPs vs LoRA Rank — {dataset.upper()}")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"./plots/{dataset}/flops_vs_rank.png", dpi=300)
    plt.close()


# ============================================================
#   PLOT 5 (Optional) — Train Loss vs Epoch Across Ranks
# ============================================================

def plot_train_loss_across_ranks(dataset, ranks):
    plt.figure(figsize=(8,6))

    for r in ranks:
        df = pd.read_csv(f"./{dataset}/r{r}/epoch_loss.csv")
        plt.plot(df["epoch"], df["train_loss"], marker="o", label=f"r={r}")

    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title(f"Train Loss vs Epoch Across Ranks — {dataset.upper()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./plots/{dataset}/train_loss_across_ranks.png", dpi=300)
    plt.close()


# ============================================================
#   COMBINED 2×2 FIGURE FOR PRESENTATION (FIXED TITLE OVERLAP)
# ============================================================

def create_combined_figure(dataset):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    img_loss   = plt.imread(f"./plots/{dataset}/loss_across_ranks.png")
    img_memory = plt.imread(f"./plots/{dataset}/memory_vs_rank.png")
    img_eval   = plt.imread(f"./plots/{dataset}/eval_loss_vs_rank.png")
    img_flops  = plt.imread(f"./plots/{dataset}/flops_vs_rank.png")

    axes[0,0].imshow(img_loss);   axes[0,0].axis("off")
    axes[0,1].imshow(img_memory); axes[0,1].axis("off")
    axes[1,0].imshow(img_eval);   axes[1,0].axis("off")
    axes[1,1].imshow(img_flops);  axes[1,1].axis("off")

    # FIX: Add more top margin so title does not overlap
    plt.subplots_adjust(top=0.88)

    # FIX: Move title higher to the top of the figure
    fig.suptitle(
        f"LoRA Cross-Rank Summary — {dataset.upper()}",
        fontsize=22,
        y=0.98  # move upward
    )

    plt.savefig(f"./plots/{dataset}/lora_crossrank_summary.png", dpi=300)
    plt.close()


# ============================================================
#                     RUN EVERYTHING
# ============================================================

print("Generating individual cross-rank plots...")
plot_loss_across_ranks(DATASET, RANKS)
plot_memory_vs_rank(DATASET, RANKS)
plot_eval_loss_vs_rank(DATASET, RANKS)
plot_flops_vs_rank(DATASET, RANKS)
plot_train_loss_across_ranks(DATASET, RANKS)

print("Generating combined 2×2 summary slide figure...")
create_combined_figure(DATASET)

print("\n✓ ALL PLOTS (AND FINAL 2×2 COMBINED FIGURE) GENERATED SUCCESSFULLY!\n")

