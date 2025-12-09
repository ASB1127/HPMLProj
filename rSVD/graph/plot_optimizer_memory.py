import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# =========================================
# CONFIG: Load optimizer memory data
# =========================================
BASE = "."  # Script runs from graph/ directory
os.makedirs("./plots", exist_ok=True)

# Helper function to convert rank to directory name
def rank_to_dir(rank):
    """Convert rank (integer) to directory name (e.g., 4 -> r4)"""
    return f"r{rank}_opt"

# Auto-detect available optimizer directories
print("Auto-detecting optimizer memory directories...")
pattern = os.path.join(BASE, "*_opt")
dirs = glob.glob(pattern)
detected_ranks = []
adam_dir = None

for d in dirs:
    dirname = os.path.basename(d)
    if dirname == "adam_opt":
        adam_dir = d
    elif dirname.endswith("_opt") and dirname.startswith("r"):
        # Try to parse rank (integer)
        try:
            rank_str = dirname[1:-4]  # Remove 'r' prefix and '_opt' suffix
            rank = int(rank_str)
            detected_ranks.append(rank)
        except ValueError:
            continue

detected_ranks = sorted(detected_ranks)
print(f"Detected ranks: {detected_ranks}")
print(f"Adam directory: {adam_dir}")

if not adam_dir or not os.path.exists(adam_dir):
    print("⚠ Warning: adam_opt directory not found!")
    exit(1)

if not detected_ranks:
    print("⚠ Warning: No r*_opt directories found!")
    exit(1)

# =========================================
# LOAD DATA
# =========================================
adam_memory = None
adam_csv = os.path.join(adam_dir, "optimizer_memory.csv")
if os.path.exists(adam_csv):
    adam_df = pd.read_csv(adam_csv)
    adam_memory = adam_df.iloc[0]['memory_mb']
    print(f"✓ Loaded Adam optimizer memory: {adam_memory:.2f} MB")
else:
    print(f"⚠ Error: {adam_csv} not found!")
    exit(1)

rsvd_data = {}
for rank in detected_ranks:
    rank_dir = os.path.join(BASE, rank_to_dir(rank))
    rsvd_csv = os.path.join(rank_dir, "optimizer_memory.csv")
    
    if os.path.exists(rsvd_csv):
        rsvd_df = pd.read_csv(rsvd_csv)
        rsvd_data[rank] = {
            'memory_mb': rsvd_df.iloc[0]['memory_mb'],
            'savings_mb': rsvd_df.iloc[0]['savings_mb'],
            'savings_pct': rsvd_df.iloc[0]['savings_pct']
        }
        print(f"✓ Loaded rSVD (rank={rank}) optimizer memory: {rsvd_data[rank]['memory_mb']:.2f} MB")
    else:
        print(f"⚠ Warning: {rsvd_csv} not found, skipping rank {rank}")

if not rsvd_data:
    print("ERROR: No rSVD optimizer memory data found!")
    exit(1)

# =========================================
# PLOT 1 — Optimizer Memory vs Rank (Bar Graph)
# =========================================
plt.figure(figsize=(10, 6))
ranks_sorted = sorted(rsvd_data.keys())
values = [rsvd_data[r]['memory_mb'] for r in ranks_sorted]
labels = [f"r{r}" for r in ranks_sorted]

# Add Adam as first bar
all_labels = ["Adam"] + labels
all_values = [adam_memory] + values
colors = ['#1f77b4'] + [plt.cm.viridis(i / len(ranks_sorted)) for i in range(len(ranks_sorted))]

plt.bar(all_labels, all_values, color=colors)
plt.xlabel("Optimizer / Rank", fontsize=12)
plt.ylabel("Optimizer State Memory (MB)", fontsize=12)
plt.title("Optimizer State Memory: Adam vs rSVD Adam", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("./plots/optimizer_memory_comparison.png", dpi=300)
plt.close()
print("✓ Saved optimizer_memory_comparison.png")

# =========================================
# PLOT 2 — Memory Savings vs Rank
# =========================================
plt.figure(figsize=(8, 5))
ranks_sorted = sorted(rsvd_data.keys())
savings_mb = [rsvd_data[r]['savings_mb'] for r in ranks_sorted]
savings_pct = [rsvd_data[r]['savings_pct'] for r in ranks_sorted]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# Bar plot for absolute savings
bars = ax1.bar([f"r{r}" for r in ranks_sorted], savings_mb, color=plt.cm.plasma(range(len(ranks_sorted))), 
                alpha=0.7, label='Savings (MB)')
ax1.set_xlabel("Rank", fontsize=12)
ax1.set_ylabel("Memory Savings (MB)", fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# Line plot for percentage savings
line = ax2.plot([f"r{r}" for r in ranks_sorted], savings_pct, marker='o', linewidth=2, 
                markersize=8, color='red', label='Savings (%)')
ax2.set_ylabel("Memory Savings (%)", fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Optimizer Memory Savings vs Rank", fontsize=14)
plt.tight_layout()
plt.savefig("./plots/optimizer_memory_savings.png", dpi=300)
plt.close()
print("✓ Saved optimizer_memory_savings.png")

# =========================================
# PLOT 3 — Optimizer Memory vs Rank (rSVD only, line plot)
# =========================================
plt.figure(figsize=(8, 5))
ranks_sorted = sorted(rsvd_data.keys())
values = [rsvd_data[r]['memory_mb'] for r in ranks_sorted]

plt.plot(ranks_sorted, values, marker="o", linewidth=2, markersize=8, color="green")
plt.xlabel("Rank", fontsize=12)
plt.ylabel("Optimizer State Memory (MB)", fontsize=12)
plt.title("rSVD Adam Optimizer Memory vs Rank", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("./plots/optimizer_memory_vs_rank.png", dpi=300)
plt.close()
print("✓ Saved optimizer_memory_vs_rank.png")

# =========================================
# PLOT 4 — Side-by-side comparison (Adam vs rSVD)
# =========================================
fig, ax = plt.subplots(figsize=(10, 6))
ranks_sorted = sorted(rsvd_data.keys())
x = range(len(ranks_sorted) + 1)  # +1 for Adam
width = 0.35

adam_values = [adam_memory] * (len(ranks_sorted) + 1)
rsvd_values = [adam_memory] + [rsvd_data[r]['memory_mb'] for r in ranks_sorted]

x_labels = ["Adam"] + [f"rSVD\nr={r}" for r in ranks_sorted]

ax.bar([i - width/2 for i in x], adam_values, width, label='Adam', color='#1f77b4')
ax.bar([i + width/2 for i in x], rsvd_values, width, label='rSVD Adam', color='#ff7f0e')

ax.set_xlabel("Optimizer", fontsize=12)
ax.set_ylabel("Optimizer State Memory (MB)", fontsize=12)
ax.set_title("Optimizer State Memory Comparison", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("./plots/optimizer_memory_side_by_side.png", dpi=300)
plt.close()
print("✓ Saved optimizer_memory_side_by_side.png")

print(f"\n🔥 All optimizer memory plots saved to: ./plots/")

