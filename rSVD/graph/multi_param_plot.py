import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# =========================================
# CONFIG: Parameters to compare
# =========================================
# For rank_fraction comparisons: ["rf0.1", "rf0.2", "rf0.3", "rf0.5"]
# For proj_interval comparisons: ["pi200", "pi500", "pi1000"]
# For dataset comparisons: ["imdb", "sst2"]
PARAM_DIRS = []  # Will be set via command line or default

# Base directory for checkpoint folders
if len(sys.argv) > 1:
    BASE_DIR = sys.argv[1]
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get parameter directories from command line (after base dir)
if len(sys.argv) > 2:
    PARAM_DIRS = sys.argv[2:]
else:
    # Default: look for common rank_fraction patterns
    print("No parameter directories specified. Looking for common patterns...")
    possible_dirs = ["rf0.1", "rf0.2", "rf0.3", "rf0.5", "bert_imdb_checkpoints", "bert_sst2_checkpoints"]
    for d in possible_dirs:
        full_path = os.path.join(BASE_DIR, d)
        if os.path.exists(full_path):
            PARAM_DIRS.append(d)
    print(f"Found directories: {PARAM_DIRS}")

if not PARAM_DIRS:
    print("ERROR: No parameter directories found!")
    print("Usage: python multi_param_plot.py [BASE_DIR] [param_dir1] [param_dir2] ...")
    print("Example: python multi_param_plot.py .. rf0.1 rf0.2 rf0.3")
    sys.exit(1)

# Create plots directory
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"\nComparing parameters: {PARAM_DIRS}")
print(f"Base directory: {BASE_DIR}")
print(f"Saving plots to: {PLOT_DIR}\n")

# =========================================
# LOAD PER-PARAM DATA
# =========================================
data = {}
memory_summary = {}
gpu_time_summary = {}

for param_dir in PARAM_DIRS:
    param_path = os.path.join(BASE_DIR, param_dir)
    
    if not os.path.exists(param_path):
        print(f"⚠ Skipping {param_dir}: directory not found")
        continue
    
    param_data = {}
    
    # Load loss data
    loss_file = os.path.join(param_path, "epoch_loss.csv")
    if os.path.exists(loss_file):
        param_data["loss"] = pd.read_csv(loss_file)
        print(f"✓ {param_dir}: Loaded loss data")
    
    # Load memory data
    mem_file = os.path.join(param_path, "epoch_memory.csv")
    if os.path.exists(mem_file):
        mem_df = pd.read_csv(mem_file)
        param_data["memory"] = mem_df
        memory_summary[param_dir] = mem_df["peak_allocated_bytes"].max() / 1e9  # GB
        print(f"✓ {param_dir}: Loaded memory data")
    
    # Load GPU time data
    gpu_file = os.path.join(param_path, "epoch_gpu_time.csv")
    if os.path.exists(gpu_file):
        gpu_df = pd.read_csv(gpu_file)
        param_data["gpu_time"] = gpu_df
        gpu_time_summary[param_dir] = gpu_df["gpu_time_ms"].mean() / 1000  # seconds
        print(f"✓ {param_dir}: Loaded GPU time data")
    
    if param_data:
        data[param_dir] = param_data

if not data:
    print("ERROR: No data found!")
    sys.exit(1)

# =========================================
# PLOT 1 — Loss curves for all parameters
# =========================================
if any("loss" in d for d in data.values()):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(range(len(PARAM_DIRS)))
    
    for i, param_dir in enumerate(PARAM_DIRS):
        if param_dir not in data or "loss" not in data[param_dir]:
            continue
        
        df = data[param_dir]["loss"]
        label = param_dir.replace("_", " ").title()
        
        if df["train_loss"].notna().any():
            plt.plot(df["epoch"], df["train_loss"], marker="o", 
                    label=f"{label} — Train", color=colors[i])
        
        if "eval_loss" in df.columns and df["eval_loss"].notna().any():
            plt.plot(df["epoch"], df["eval_loss"], marker="s", 
                    linestyle="--", label=f"{label} — Eval", 
                    color=colors[i], alpha=0.7)
    
    plt.title("Loss per Epoch Across Parameters", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "loss_across_params.png"), dpi=300)
    plt.close()
    print("✓ Saved loss_across_params.png")

# =========================================
# PLOT 2 — Peak GPU Memory per Epoch (all params)
# =========================================
if any("memory" in d for d in data.values()):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(range(len(PARAM_DIRS)))
    
    for i, param_dir in enumerate(PARAM_DIRS):
        if param_dir not in data or "memory" not in data[param_dir]:
            continue
        
        df = data[param_dir]["memory"]
        label = param_dir.replace("_", " ").title()
        plt.plot(df["epoch"], df["peak_allocated_bytes"] / 1e9, 
                marker="o", label=label, color=colors[i])
    
    plt.title("Peak GPU Memory per Epoch Across Parameters", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Peak Memory (GB)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "peak_memory_per_epoch_across_params.png"), dpi=300)
    plt.close()
    print("✓ Saved peak_memory_per_epoch_across_params.png")

# =========================================
# PLOT 3 — Peak GPU Memory vs Parameter
# =========================================
if memory_summary:
    plt.figure(figsize=(8, 5))
    params = list(memory_summary.keys())
    values = list(memory_summary.values())
    
    plt.bar([p.replace("_", " ").title() for p in params], values, 
            color=plt.cm.viridis(range(len(params))))
    plt.ylabel("Peak Memory (GB)", fontsize=12)
    plt.title("Peak GPU Memory vs Parameter", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "peak_memory_vs_param.png"), dpi=300)
    plt.close()
    print("✓ Saved peak_memory_vs_param.png")

# =========================================
# PLOT 4 — Average GPU Time vs Parameter
# =========================================
if gpu_time_summary:
    plt.figure(figsize=(8, 5))
    params = list(gpu_time_summary.keys())
    values = list(gpu_time_summary.values())
    
    plt.bar([p.replace("_", " ").title() for p in params], values, 
            color=plt.cm.plasma(range(len(params))))
    plt.ylabel("Average GPU Time per Epoch (seconds)", fontsize=12)
    plt.title("GPU Time vs Parameter", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gpu_time_vs_param.png"), dpi=300)
    plt.close()
    print("✓ Saved gpu_time_vs_param.png")

# =========================================
# PLOT 5 — GPU Time per Epoch (all params)
# =========================================
if any("gpu_time" in d for d in data.values()):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(range(len(PARAM_DIRS)))
    
    for i, param_dir in enumerate(PARAM_DIRS):
        if param_dir not in data or "gpu_time" not in data[param_dir]:
            continue
        
        df = data[param_dir]["gpu_time"]
        label = param_dir.replace("_", " ").title()
        plt.plot(df["epoch"], df["gpu_time_ms"] / 1000, 
                marker="o", label=label, color=colors[i])
    
    plt.title("GPU Time per Epoch Across Parameters", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("GPU Time (seconds)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gpu_time_per_epoch_across_params.png"), dpi=300)
    plt.close()
    print("✓ Saved gpu_time_per_epoch_across_params.png")

print(f"\n🔥 All multi-parameter comparison plots saved to: {PLOT_DIR}/")

