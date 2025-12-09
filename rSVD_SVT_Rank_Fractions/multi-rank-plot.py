import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# =========================================
# CONFIG: all rank_fractions to load
# =========================================
# Update this list with the rank_fractions you've run
# The script will look for directories like r0_1, r0_3, r0_5, etc.
RANK_FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5]  # Update based on your runs
BASE = "./sst/graph" # or ./imdbb/graph          

os.makedirs(BASE + "/plots", exist_ok=True)

# Helper function to convert rank_fraction to directory name
def rank_fraction_to_dir(rf):
    """Convert rank_fraction (float) to directory name (e.g., 0.3 -> r0_3)"""
    return f"r{rf}".replace(".", "_")

# Verify we're in the right directory (should have r* subdirectories)
if not glob.glob(os.path.join(BASE, "r*")):
    print(f"⚠ Warning: No r* directories found in '{BASE}'. Make sure you're running this script from the graph/ directory.")
    print(f"   Expected to find directories like: r0_01, r0_05, etc.")

# Auto-detect available rank_fraction directories if RANK_FRACTIONS is empty
if not RANK_FRACTIONS:
    print("Auto-detecting rank_fraction directories...")
    pattern = os.path.join(BASE, "r*")
    dirs = glob.glob(pattern)
    detected = []
    for d in dirs:
        dirname = os.path.basename(d)
        if dirname.startswith("r") and os.path.isdir(d):
            # Try to parse back to float
            try:
                rf_str = dirname[1:].replace("_", ".")
                rf = float(rf_str)
                detected.append(rf)
            except ValueError:
                continue
    RANK_FRACTIONS = sorted(detected)
    print(f"Detected rank_fractions: {RANK_FRACTIONS}")

# Storage for comparison plots
memory_summary = {}
flops_summary = {}
program_mem_summary = {}
forward_mem_summary = {}
forward_flops_summary = {}

# =========================================
# LOAD PER-RANK-FRACTION DATA
# =========================================
data = {}

for rf in RANK_FRACTIONS:
    rank_dir = os.path.join(BASE, rank_fraction_to_dir(rf))
    
    if not os.path.exists(rank_dir):
        print(f"⚠ Skipping rank_fraction {rf}: directory {rank_dir} not found")
        continue
    
    print(f"Loading data from {rank_dir}...")
    
    try:
        data[rf] = {
            "epoch_mem": pd.read_csv(f"{rank_dir}/epoch_peak_memory.csv"),
            "flops": pd.read_csv(f"{rank_dir}/flops_profiler_stats.csv"),
            "total_mem": pd.read_csv(f"{rank_dir}/total_program_memory.csv"),
            "loss": pd.read_csv(f"{rank_dir}/epoch_loss.csv"),
        }
        
        # Load forward pass data if available
        forward_pass_dir = os.path.join(rank_dir, "forward_pass")
        if os.path.exists(forward_pass_dir):
            if os.path.exists(f"{forward_pass_dir}/peak_forward_memory.csv"):
                data[rf]["forward_mem"] = pd.read_csv(f"{forward_pass_dir}/peak_forward_memory.csv")
            if os.path.exists(f"{forward_pass_dir}/forward_flops.csv"):
                data[rf]["forward_flops"] = pd.read_csv(f"{forward_pass_dir}/forward_flops.csv")
        
        # Summaries for comparison plots
        memory_summary[rf] = data[rf]["epoch_mem"]["peak_memory_bytes"].max() / 1e9
        
        # Extract FLOPs (handle case where metric might not exist)
        flops_row = data[rf]["flops"].loc[data[rf]["flops"]["metric"] == "step_flops"]
        if not flops_row.empty:
            flops_summary[rf] = flops_row["value"].item()
        else:
            print(f"⚠ Warning: step_flops not found for rank_fraction {rf}")
            
        # Extract total program memory
        total_mem_row = data[rf]["total_mem"].loc[data[rf]["total_mem"]["metric"] == "program_total_peak_memory"]
        if not total_mem_row.empty:
            program_mem_summary[rf] = total_mem_row["value_bytes"].item() / 1e9
        else:
            print(f"⚠ Warning: program_total_peak_memory not found for rank_fraction {rf}")
        
        if "forward_mem" in data[rf]:
            forward_mem_summary[rf] = (
                data[rf]["forward_mem"]
                .loc[data[rf]["forward_mem"]["metric"] == "peak_memory", "value_bytes"]
                .item() / 1e9
            )
        
        if "forward_flops" in data[rf]:
            forward_flops_summary[rf] = (
                data[rf]["forward_flops"]
                .loc[data[rf]["forward_flops"]["metric"] == "forward_flops", "value"]
                .item()
            )
        
        print(f"✓ Loaded data for rank_fraction {rf}")
    except Exception as e:
        print(f"⚠ Error loading data for rank_fraction {rf}: {e}")
        continue

if not data:
    print("ERROR: No data found! Make sure you've run training and have CSV files in the graph directories.")
    print(f"Expected directories: {[rank_fraction_to_dir(rf) for rf in RANK_FRACTIONS]}")
    exit(1)

# =========================================
# PLOT 1 — Loss curves for all rank_fractions
# =========================================
plt.figure(figsize=(8,6))
for rf in RANK_FRACTIONS:
    if rf not in data:
        continue
    df = data[rf]["loss"]
    label = f"Rank Fraction {rf}"
    # Convert empty strings to NaN for numeric columns
    df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
    df["eval_loss"] = pd.to_numeric(df["eval_loss"], errors="coerce")
    
    if df["train_loss"].notna().any():
        plt.plot(df["epoch"], df["train_loss"], marker="o", label=f"{label} — Train Loss")
    if df["eval_loss"].notna().any():
        plt.plot(df["epoch"], df["eval_loss"], marker="o", linestyle="--", label=f"{label} — Eval Loss")

plt.title("Loss per Epoch Across rSVD Rank Fractions", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(BASE + "/plots/loss_across_rank_fractions.png", dpi=300)
plt.close()
print("✓ Saved loss_across_rank_fractions.png")

# =========================================
# PLOT 2 — Peak GPU Memory per Epoch (all rank_fractions)
# =========================================
plt.figure(figsize=(8,6))
for rf in RANK_FRACTIONS:
    if rf not in data:
        continue
    df = data[rf]["epoch_mem"]
    plt.plot(df["epoch"], df["peak_memory_bytes"] / 1e9, marker="o", label=f"Rank Fraction {rf}")

plt.title("Peak GPU Memory per Epoch Across Rank Fractions", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Peak Memory (GB)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(BASE + "/plots/peak_memory_per_epoch_across_rank_fractions.png", dpi=300)
plt.close()
print("✓ Saved peak_memory_per_epoch_across_rank_fractions.png")

# =========================================
# PLOT 3 — FLOPs vs Rank Fraction
# =========================================
if flops_summary:
    plt.figure(figsize=(8,5))
    rfs = sorted(flops_summary.keys())
    plt.plot(rfs, [flops_summary[rf] / 1e12 for rf in rfs], marker="o", linewidth=2, markersize=8)
    plt.xlabel("Rank Fraction")
    plt.ylabel("FLOPs per Step (TFLOPs)")
    plt.title("Compute Cost vs rSVD Rank Fraction")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(BASE + "/plots/flops_vs_rank_fraction.png", dpi=300)
    plt.close()
    print("✓ Saved flops_vs_rank_fraction.png")

# =========================================
# PLOT 4 — Peak GPU Memory vs Rank Fraction
# =========================================
if memory_summary:
    plt.figure(figsize=(8,5))
    rfs = sorted(memory_summary.keys())
    plt.plot(rfs, [memory_summary[rf] for rf in rfs], marker="o", linewidth=2, markersize=8)
    plt.xlabel("Rank Fraction")
    plt.ylabel("Peak Memory (GB)")
    plt.title("Peak GPU Memory vs rSVD Rank Fraction")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(BASE + "/plots/peak_memory_vs_rank_fraction.png", dpi=300)
    plt.close()
    print("✓ Saved peak_memory_vs_rank_fraction.png")

# =========================================
# PLOT 5 — Total Program Memory vs Rank Fraction
# =========================================
if program_mem_summary:
    plt.figure(figsize=(8,5))
    rfs = sorted(program_mem_summary.keys())
    plt.plot(rfs, [program_mem_summary[rf] for rf in rfs], marker="o", linewidth=2, markersize=8)
    plt.xlabel("Rank Fraction")
    plt.ylabel("Memory (GB)")
    plt.title("Total Program Peak Memory vs Rank Fraction")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(BASE + "/plots/total_program_memory_vs_rank_fraction.png", dpi=300)
    plt.close()
    print("✓ Saved total_program_memory_vs_rank_fraction.png")

# =========================================
# PLOT 6 — Forward Pass Memory vs Rank Fraction (if available)
# =========================================
if forward_mem_summary:
    plt.figure(figsize=(8,5))
    rfs = sorted(forward_mem_summary.keys())
    plt.plot(rfs, [forward_mem_summary[rf] for rf in rfs], marker="o", linewidth=2, markersize=8, color="green")
    plt.xlabel("Rank Fraction")
    plt.ylabel("Peak Forward Memory (GB)")
    plt.title("Forward Pass Peak Memory vs Rank Fraction")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(BASE + "/plots/forward_memory_vs_rank_fraction.png", dpi=300)
    plt.close()
    print("✓ Saved forward_memory_vs_rank_fraction.png")

# =========================================
# PLOT 7 — Forward Pass FLOPs vs Rank Fraction (if available)
# =========================================
if forward_flops_summary:
    plt.figure(figsize=(8,5))
    rfs = sorted(forward_flops_summary.keys())
    plt.plot(rfs, [forward_flops_summary[rf] / 1e9 for rf in rfs], marker="o", linewidth=2, markersize=8, color="purple")
    plt.xlabel("Rank Fraction")
    plt.ylabel("Forward FLOPs (GFLOPs)")
    plt.title("Forward Pass FLOPs vs Rank Fraction")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(BASE + "/plots/forward_flops_vs_rank_fraction.png", dpi=300)
    plt.close()
    print("✓ Saved forward_flops_vs_rank_fraction.png")

print(f"\n🔥 All multi-rank-fraction rSVD plots saved to: " + BASE +  "/plots/")

