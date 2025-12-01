import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "/Users/amitbal/Downloads/HPMLProj/Lora/graph"
FIG_DIR  = "/Users/amitbal/Downloads/HPMLProj/Lora/graph/plots/run2"

os.makedirs(FIG_DIR, exist_ok=True)

ranks = [4, 8, 16, 64, 128, 700]


# ======================
# Helpers
# ======================

def load_loss_csv(rank):
    path = os.path.join(DATA_DIR, f"r{rank}", "epoch_loss.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["epoch"])  # remove trailing ",,"
    return df


def load_epoch_peak_memory(rank):
    path = os.path.join(DATA_DIR, f"r{rank}", "epoch_peak_memory.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["epoch"])
    return df


def load_total_program_memory(rank):
    path = os.path.join(DATA_DIR, f"r{rank}", "total_program_memory.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # find the row where metric == "program_total_peak_memory"
    row = df[df["metric"] == "program_total_peak_memory"]
    if len(row) == 0:
        return None
    return row["value_bytes"].iloc[0]


def load_forward_flops(rank):
    path = os.path.join(DATA_DIR, f"r{rank}", "forward_pass", "forward_flops.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    row = df[df["metric"] == "forward_flops"]
    if len(row) == 0:
        return None
    return row["value"].iloc[0]


def load_forward_peak_memory(rank):
    path = os.path.join(DATA_DIR, f"r{rank}", "forward_pass", "peak_forward_memory.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    row = df[df["metric"] == "peak_memory"]
    if len(row) == 0:
        return None
    return row["value_bytes"].iloc[0]


# ============================================================
# 1. LOSS vs EPOCH
# ============================================================

plt.figure(figsize=(7,5))

for r in ranks:
    df = load_loss_csv(r)
    if df is None:
        continue

    plt.plot(df["epoch"], df["train_loss"], marker="o", label=f"r={r}")

plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "loss_vs_epoch.png"))
plt.show()


# ============================================================
# 2. PEAK VRAM vs EPOCH (PER-RANK)
# ============================================================

plt.figure(figsize=(7,5))

for r in ranks:
    df = load_epoch_peak_memory(r)
    if df is None:
        continue

    vals_gb = df["peak_memory_bytes"] / (1024**3)

    plt.plot(df["epoch"], vals_gb, marker="o", label=f"r={r}")

plt.title("Peak GPU Memory per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Peak Memory (GB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "peak_vram_vs_epoch.png"))
plt.show()


# ============================================================
# 3. PEAK VRAM vs RANK (FINAL PEAK)
# ============================================================

peak_mem_gb = []
plot_ranks = [r for r in ranks if r != 700] 
for r in plot_ranks:
    val = load_total_program_memory(r)
    if val is None:
        peak_mem_gb.append(None)
    else:
        peak_mem_gb.append(val / (1024**3))  # bytes → GB

plt.figure(figsize=(7,5))
plt.plot(plot_ranks, peak_mem_gb, marker="o")
plt.title("Final Peak GPU Memory vs LoRA Rank")
plt.xlabel("LoRA Rank")
plt.ylabel("Peak Memory (GB)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "peak_vram_vs_rank.png"))
plt.show()


# ============================================================
# 4. FLOPs vs RANK
# ============================================================

forward_flops = []

for r in ranks:
    val = load_forward_flops(r)
    forward_flops.append(val)

plt.figure(figsize=(7,5))
plt.plot(ranks, forward_flops, marker="o")
plt.yscale("log")
plt.title("Forward FLOPs vs LoRA Rank")
plt.xlabel("LoRA Rank")
plt.ylabel("FLOPs (log scale)")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "flops_vs_rank.png"))
plt.show()

# ============================================================
# 5. FORWARD PASS VRAM vs RANK  (Option B: skip missing)
# ============================================================

valid_ranks = []
valid_mems = []

for r in ranks:
    val = load_forward_peak_memory(r)
    if val is None:
        print(f"[INFO] forward_pass missing for r={r} — skipping in plot.")
        continue

    valid_ranks.append(r)
    valid_mems.append(val / (1024**3))  # convert bytes → GB

plt.figure(figsize=(7,5))
plt.bar([str(r) for r in valid_ranks], valid_mems)
plt.title("Forward Pass Memory vs LoRA Rank")
plt.xlabel("LoRA Rank")
plt.ylabel("Peak Forward Memory (GB)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "forward_mem_vs_rank.png"))
plt.show()



