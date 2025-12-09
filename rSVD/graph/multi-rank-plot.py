import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# =========================================
# CONFIG: all ranks to load
# =========================================
# Update this list with the ranks you've run
# The script will look for directories like r4, r8, r16, etc.
RANKS = [4, 8, 16, 64, 128]  # Update based on your runs
BASE = "."          # <--- Script runs from graph/ directory, so BASE is "."

os.makedirs("./plots", exist_ok=True)

# Helper function to convert rank to directory name
def rank_to_dir(rank):
    """Convert rank (integer) to directory name (e.g., 4 -> r4)"""
    return f"r{rank}"

# Verify we're in the right directory (should have r* subdirectories)
if not glob.glob(os.path.join(BASE, "r*")):
    print(f"⚠ Warning: No r* directories found in '{BASE}'. Make sure you're running this script from the graph/ directory.")
    print(f"   Expected to find directories like: r4, r8, r16, etc.")

# Auto-detect available rank directories if RANKS is empty
if not RANKS:
    print("Auto-detecting rank directories...")
    pattern = os.path.join(BASE, "r*")
    dirs = glob.glob(pattern)
    detected = []
    for d in dirs:
        dirname = os.path.basename(d)
        if dirname.startswith("r") and os.path.isdir(d) and not dirname.endswith("_opt"):
            # Try to parse back to integer
            try:
                rank = int(dirname[1:])  # Remove 'r' prefix
                detected.append(rank)
            except ValueError:
                continue
    RANKS = sorted(detected)
    print(f"Detected ranks: {RANKS}")

# Storage for comparison plots
memory_summary = {}
flops_summary = {}
epoch_flops_summary = {}
program_mem_summary = {}
forward_mem_summary = {}
forward_flops_summary = {}

# =========================================
# LOAD PER-RANK DATA
# =========================================
data = {}

for rank in RANKS:
    rank_dir = os.path.join(BASE, rank_to_dir(rank))
    
    if not os.path.exists(rank_dir):
        print(f"⚠ Skipping rank {rank}: directory {rank_dir} not found")
        continue
    
    print(f"Loading data from {rank_dir}...")
    
    try:
        data[rank] = {
            "epoch_mem": pd.read_csv(f"{rank_dir}/epoch_peak_memory.csv"),
            "flops": pd.read_csv(f"{rank_dir}/flops_profiler_stats.csv"),
            "total_mem": pd.read_csv(f"{rank_dir}/total_program_memory.csv"),
            "loss": pd.read_csv(f"{rank_dir}/epoch_loss.csv"),
        }
        
        # Load forward pass data if available
        forward_pass_dir = os.path.join(rank_dir, "forward_pass")
        if os.path.exists(forward_pass_dir):
            if os.path.exists(f"{forward_pass_dir}/peak_forward_memory.csv"):
                data[rank]["forward_mem"] = pd.read_csv(f"{forward_pass_dir}/peak_forward_memory.csv")
            if os.path.exists(f"{forward_pass_dir}/forward_flops.csv"):
                data[rank]["forward_flops"] = pd.read_csv(f"{forward_pass_dir}/forward_flops.csv")
        
        # Summaries for comparison plots
        memory_summary[rank] = data[rank]["epoch_mem"]["peak_memory_bytes"].max() / 1e9
        
        # Extract FLOPs (handle case where metric might not exist)
        flops_row = data[rank]["flops"].loc[data[rank]["flops"]["metric"] == "step_flops"]
        if not flops_row.empty:
            flops_summary[rank] = flops_row["value"].item()
        else:
            print(f"⚠ Warning: step_flops not found for rank {rank}")
        
        # Extract epoch FLOPs
        epoch_flops_row = data[rank]["flops"].loc[data[rank]["flops"]["metric"] == "epoch_flops"]
        if not epoch_flops_row.empty:
            epoch_flops_summary[rank] = epoch_flops_row["value"].item()
        else:
            print(f"⚠ Warning: epoch_flops not found for rank {rank}")
            
        # Extract total program memory
        total_mem_row = data[rank]["total_mem"].loc[data[rank]["total_mem"]["metric"] == "program_total_peak_memory"]
        if not total_mem_row.empty:
            program_mem_summary[rank] = total_mem_row["value_bytes"].item() / 1e9
        else:
            print(f"⚠ Warning: program_total_peak_memory not found for rank {rank}")
        
        if "forward_mem" in data[rank]:
            forward_mem_summary[rank] = (
                data[rank]["forward_mem"]
                .loc[data[rank]["forward_mem"]["metric"] == "peak_memory", "value_bytes"]
                .item() / 1e9
            )
        
        if "forward_flops" in data[rank]:
            forward_flops_summary[rank] = (
                data[rank]["forward_flops"]
                .loc[data[rank]["forward_flops"]["metric"] == "forward_flops", "value"]
                .item()
            )
        
        print(f"✓ Loaded data for rank {rank}")
    except Exception as e:
        print(f"⚠ Error loading data for rank {rank}: {e}")
        continue

if not data:
    print("ERROR: No data found! Make sure you've run training and have CSV files in the graph directories.")
    print(f"Expected directories: {[rank_to_dir(r) for r in RANKS]}")
    exit(1)

# =========================================
# PLOT 1 — Loss curves for all ranks
# =========================================
plt.figure(figsize=(8,6))
for rank in RANKS:
    if rank not in data:
        continue
    df = data[rank]["loss"]
    label = f"Rank {rank}"
    # Convert empty strings to NaN for numeric columns
    df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
    df["eval_loss"] = pd.to_numeric(df["eval_loss"], errors="coerce")
    
    if df["train_loss"].notna().any():
        plt.plot(df["epoch"], df["train_loss"], marker="o", label=f"{label} — Train Loss")
    if df["eval_loss"].notna().any():
        plt.plot(df["epoch"], df["eval_loss"], marker="o", linestyle="--", label=f"{label} — Eval Loss")

plt.title("Loss per Epoch Across rSVD Ranks", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/loss_across_ranks.png", dpi=300)
plt.close()
print("✓ Saved loss_across_ranks.png")

# =========================================
# PLOT 2 — Peak GPU Memory per Epoch (all ranks)
# =========================================
plt.figure(figsize=(8,6))
for rank in RANKS:
    if rank not in data:
        continue
    df = data[rank]["epoch_mem"]
    plt.plot(df["epoch"], df["peak_memory_bytes"] / 1e9, marker="o", label=f"Rank {rank}")

plt.title("Peak GPU Memory per Epoch Across Ranks", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Peak Memory (GB)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/peak_memory_per_epoch_across_ranks.png", dpi=300)
plt.close()
print("✓ Saved peak_memory_per_epoch_across_ranks.png")

# =========================================
# PLOT 3 — FLOPs per Step vs Rank (Bar Graph)
# =========================================
if flops_summary:
    plt.figure(figsize=(8,5))
    ranks_sorted = sorted(flops_summary.keys())
    values = [flops_summary[r] / 1e12 for r in ranks_sorted]
    labels = [f"r{r}" for r in ranks_sorted]
    plt.bar(labels, values, color=plt.cm.viridis(range(len(ranks_sorted))))
    plt.xlabel("Rank")
    plt.ylabel("FLOPs per Step (TFLOPs)")
    plt.title("Compute Cost per Step vs rSVD Rank")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("./plots/flops_vs_rank.png", dpi=300)
    plt.close()
    print("✓ Saved flops_vs_rank.png")

# =========================================
# PLOT 3b — Epoch FLOPs vs Rank (Bar Graph)
# =========================================
if epoch_flops_summary:
    plt.figure(figsize=(8,5))
    ranks_sorted = sorted(epoch_flops_summary.keys())
    values = [epoch_flops_summary[r] / 1e15 for r in ranks_sorted]
    labels = [f"r{r}" for r in ranks_sorted]
    plt.bar(labels, values, color=plt.cm.plasma(range(len(ranks_sorted))))
    plt.xlabel("Rank")
    plt.ylabel("FLOPs per Epoch (PFLOPs)")
    plt.title("Compute Cost per Epoch vs rSVD Rank")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("./plots/epoch_flops_vs_rank.png", dpi=300)
    plt.close()
    print("✓ Saved epoch_flops_vs_rank.png")

# =========================================
# PLOT 4 — Peak GPU Memory vs Rank (Bar Graph)
# =========================================
if memory_summary:
    plt.figure(figsize=(8,5))
    ranks_sorted = sorted(memory_summary.keys())
    values = [memory_summary[r] for r in ranks_sorted]
    labels = [f"r{r}" for r in ranks_sorted]
    plt.bar(labels, values, color=plt.cm.coolwarm(range(len(ranks_sorted))))
    plt.xlabel("Rank")
    plt.ylabel("Peak Memory (GB)")
    plt.title("Peak GPU Memory vs rSVD Rank")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("./plots/peak_memory_vs_rank.png", dpi=300)
    plt.close()
    print("✓ Saved peak_memory_vs_rank.png")

# =========================================
# PLOT 5 — Total Program Memory vs Rank
# =========================================
if program_mem_summary:
    plt.figure(figsize=(8,5))
    ranks_sorted = sorted(program_mem_summary.keys())
    plt.plot(ranks_sorted, [program_mem_summary[r] for r in ranks_sorted], marker="o", linewidth=2, markersize=8)
    plt.xlabel("Rank")
    plt.ylabel("Memory (GB)")
    plt.title("Total Program Peak Memory vs Rank")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("./plots/total_program_memory_vs_rank.png", dpi=300)
    plt.close()
    print("✓ Saved total_program_memory_vs_rank.png")

# =========================================
# PLOT 6 — Forward Pass Memory vs Rank (if available)
# =========================================
if forward_mem_summary:
    plt.figure(figsize=(8,5))
    ranks_sorted = sorted(forward_mem_summary.keys())
    plt.plot(ranks_sorted, [forward_mem_summary[r] for r in ranks_sorted], marker="o", linewidth=2, markersize=8, color="green")
    plt.xlabel("Rank")
    plt.ylabel("Peak Forward Memory (GB)")
    plt.title("Forward Pass Peak Memory vs Rank")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("./plots/forward_memory_vs_rank.png", dpi=300)
    plt.close()
    print("✓ Saved forward_memory_vs_rank.png")

# =========================================
# PLOT 7 — Forward Pass FLOPs vs Rank (if available)
# =========================================
if forward_flops_summary:
    plt.figure(figsize=(8,5))
    ranks_sorted = sorted(forward_flops_summary.keys())
    plt.plot(ranks_sorted, [forward_flops_summary[r] / 1e9 for r in ranks_sorted], marker="o", linewidth=2, markersize=8, color="purple")
    plt.xlabel("Rank")
    plt.ylabel("Forward FLOPs (GFLOPs)")
    plt.title("Forward Pass FLOPs vs Rank")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("./plots/forward_flops_vs_rank.png", dpi=300)
    plt.close()
    print("✓ Saved forward_flops_vs_rank.png")

print(f"\n🔥 All multi-rank rSVD plots saved to: ./plots/")

