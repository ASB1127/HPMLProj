import pandas as pd
import matplotlib.pyplot as plt
import os

# Make sure plot directory exists
os.makedirs("./plots", exist_ok=True)

# ===============================
# Load CSV Files
# ===============================
epoch_mem = pd.read_csv("./epoch_peak_memory.csv")
flops = pd.read_csv("./flops_profiler_stats.csv")
total_mem = pd.read_csv("./total_program_memory.csv")

# ===============================
# PLOT 1 — Peak GPU Memory per Epoch
# ===============================
plt.figure(figsize=(8,5))
plt.plot(
    epoch_mem["epoch"],
    epoch_mem["peak_memory_bytes"] / 1e9,    # Convert to GB
    marker="o",
    linewidth=2,
    color="#1f77b4"
)
plt.title("LoRA Fine-Tuning: Peak GPU Memory per Epoch", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Peak Memory (GB)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)



plt.tight_layout()
plt.savefig("./plots/lora_memory_across_epochs.png", dpi=300)
plt.close()


# ===============================
# PLOT 2 — FLOPs per Step vs FLOPs per Epoch
# ===============================
step_flops = flops.loc[flops["metric"] == "step_flops", "value"].item()
epoch_flops = flops.loc[flops["metric"] == "epoch_flops", "value"].item()

plt.figure(figsize=(8,5))
plt.bar(
    ["Step FLOPs (TFLOPs)", "Epoch FLOPs (PFLOPs)"],
    [step_flops / 1e12, epoch_flops / 1e15],
    color=["#2ca02c", "#ff7f0e"]
)

plt.title("Compute Cost of LoRA Fine-Tuning", fontsize=14)
plt.ylabel("FLOPs", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)


plt.tight_layout()
plt.savefig("./plots/lora_compute_cost.png", dpi=300)
plt.close()


# ===============================
# PLOT 3 — Epoch Peak Memory vs Total Program Memory
# ===============================
epoch_peak_gb = epoch_mem["peak_memory_bytes"].max() / 1e9
program_peak_gb = total_mem.loc[
    total_mem["metric"] == "program_total_peak_memory", "value_bytes"
].item() / 1e9

plt.figure(figsize=(8,5))
plt.bar(
    ["Peak per Epoch", "Total Program Peak"],
    [epoch_peak_gb, program_peak_gb],
    color=["#9467bd", "#d62728"]
)

plt.title("GPU Memory Footprint of LoRA Fine-Tuning", fontsize=14)
plt.ylabel("Memory (GB)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)



plt.tight_layout()
plt.savefig("./plots/lora_memory_breakdown.png", dpi=300)
plt.close()

print("🔥 All LoRA baseline plots saved to: ./plots/")
