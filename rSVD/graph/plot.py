"""
Consolidated plotting script for rSVD experiments.

This script generates all plots for rSVD experiments, including:
- Multi-rank comparisons (loss, memory, FLOPs)
- Memory breakdown analysis
- Optimizer memory comparisons
- LoRA vs rSVD comparisons

Run this script from the graph/ directory.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np
from typing import Dict, Optional
from pathlib import Path

# =========================================
# CONFIGURATION
# =========================================
RANKS = [4, 8, 16, 64, 128]  # Ranks to plot
BASE = "."  # Base directory (should be graph/ when script runs from graph/)
OUTPUT_DIR = "./plots"  # Output directory for plots
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paths for comparison plots (relative to graph/ directory)
LORA_BASE_PATH = "../../Lora/graph/sst2"
RSVD_BASE_PATH = "."
RSVD_WEIGHT_REDUCTION_BASE_PATH = "../../rSVD_Weight_Reduction/graph/sst2"
RSVD_SVT_BASE_PATH = "../../rSVD_SVT_Fixed_Rank/sst/graph"


# =========================================
# HELPER FUNCTIONS
# =========================================
def rank_to_dir(rank):
    """Convert rank (integer) to directory name (e.g., 4 -> r4)"""
    return f"r{rank}"


def parse_rank_from_filename(filename: str) -> Optional[int]:
    """Extract rank from filename like 'memory_breakdown_r4.csv' or 'memory_breakdown.csv'."""
    match = re.search(r'r(\d+)', filename)
    if match:
        return int(match.group(1))
    return None  # Baseline (Adam)


def extract_component_value(df: pd.DataFrame, component_name: str) -> float:
    """Extract memory value for a specific component."""
    row = df[df['component'] == component_name]
    if len(row) > 0:
        return row.iloc[0]['memory_mb']
    return 0.0


# =========================================
# MULTI-RANK PLOTS
# =========================================
def plot_multi_rank_plots():
    """Generate multi-rank comparison plots."""
    print("\n" + "=" * 80)
    print("Generating Multi-Rank Plots")
    print("=" * 80)
    
    # Auto-detect available rank directories if RANKS is empty
    ranks_to_plot = RANKS
    if not ranks_to_plot:
        print("Auto-detecting rank directories...")
        pattern = os.path.join(BASE, "r*")
        dirs = glob.glob(pattern)
        detected = []
        for d in dirs:
            dirname = os.path.basename(d)
            if dirname.startswith("r") and os.path.isdir(d) and not dirname.endswith("_opt"):
                try:
                    rank = int(dirname[1:])
                    detected.append(rank)
                except ValueError:
                    continue
        ranks_to_plot = sorted(detected)
        print(f"Detected ranks: {ranks_to_plot}")
    
    # Storage for comparison plots
    memory_summary = {}
    flops_summary = {}
    epoch_flops_summary = {}
    program_mem_summary = {}
    forward_mem_summary = {}
    forward_flops_summary = {}
    
    # Load per-rank data
    data = {}
    for rank in ranks_to_plot:
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
            
            # Load training accuracy data if available
            train_acc_path = f"{rank_dir}/epoch_train_accuracy.csv"
            if os.path.exists(train_acc_path):
                data[rank]["train_acc"] = pd.read_csv(train_acc_path)
            
            # Load forward pass data if available
            forward_pass_dir = os.path.join(rank_dir, "forward_pass")
            if os.path.exists(forward_pass_dir):
                if os.path.exists(f"{forward_pass_dir}/peak_forward_memory.csv"):
                    data[rank]["forward_mem"] = pd.read_csv(f"{forward_pass_dir}/peak_forward_memory.csv")
                if os.path.exists(f"{forward_pass_dir}/forward_flops.csv"):
                    data[rank]["forward_flops"] = pd.read_csv(f"{forward_pass_dir}/forward_flops.csv")
            
            # Summaries for comparison plots
            memory_summary[rank] = data[rank]["epoch_mem"]["peak_memory_bytes"].max() / 1e9
            
            flops_row = data[rank]["flops"].loc[data[rank]["flops"]["metric"] == "step_flops"]
            if not flops_row.empty:
                flops_summary[rank] = flops_row["value"].item()
            
            epoch_flops_row = data[rank]["flops"].loc[data[rank]["flops"]["metric"] == "epoch_flops"]
            if not epoch_flops_row.empty:
                epoch_flops_summary[rank] = epoch_flops_row["value"].item()
            
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
        print("ERROR: No data found!")
        return
    
    # Plot 1: Training Loss curves
    plt.figure(figsize=(8, 6))
    for rank in ranks_to_plot:
        if rank not in data:
            continue
        df = data[rank]["loss"]
        label = f"Rank {rank}"
        df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
        if df["train_loss"].notna().any():
            plt.plot(df["epoch"], df["train_loss"], marker="o", label=label, linewidth=2, markersize=6)
    plt.title("Training Loss per Epoch Across rSVD Ranks", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_loss_across_ranks.png", dpi=300)
    plt.close()
    print("✓ Saved training_loss_across_ranks.png")
    
    # Plot 1a: Training Accuracy curves
    plt.figure(figsize=(8, 6))
    for rank in ranks_to_plot:
        if rank not in data or "train_acc" not in data[rank]:
            continue
        df = data[rank]["train_acc"]
        label = f"Rank {rank}"
        df["train_accuracy"] = pd.to_numeric(df["train_accuracy"], errors="coerce")
        if df["train_accuracy"].notna().any():
            plt.plot(df["epoch"], df["train_accuracy"], marker="o", label=label, linewidth=2, markersize=6)
    plt.title("Training Accuracy per Epoch Across rSVD Ranks", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_accuracy_across_ranks.png", dpi=300)
    plt.close()
    print("✓ Saved training_accuracy_across_ranks.png")
    
    # Plot 1c: Average Validation Loss vs Rank
    avg_eval_loss_summary = {}
    for rank in ranks_to_plot:
        if rank not in data:
            continue
        df = data[rank]["loss"]
        df["eval_loss"] = pd.to_numeric(df["eval_loss"], errors="coerce")
        if df["eval_loss"].notna().any():
            avg_eval_loss_summary[rank] = df["eval_loss"].mean()
    
    if avg_eval_loss_summary:
        plt.figure(figsize=(8, 5))
        ranks_sorted = sorted(avg_eval_loss_summary.keys())
        values = [avg_eval_loss_summary[r] for r in ranks_sorted]
        labels = [f"r{r}" for r in ranks_sorted]
        plt.bar(labels, values, color=plt.cm.Set3(range(len(ranks_sorted))))
        plt.xlabel("Rank")
        plt.ylabel("Average Validation Loss")
        plt.title("Average Validation Loss vs rSVD Rank")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/avg_validation_loss_vs_rank.png", dpi=300)
        plt.close()
        print("✓ Saved avg_validation_loss_vs_rank.png")
    
    # Plot 1b: Loss curves (train + eval)
    plt.figure(figsize=(8, 6))
    for rank in ranks_to_plot:
        if rank not in data:
            continue
        df = data[rank]["loss"]
        label = f"Rank {rank}"
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
    plt.savefig(f"{OUTPUT_DIR}/loss_across_ranks.png", dpi=300)
    plt.close()
    print("✓ Saved loss_across_ranks.png")
    
    # Plot 2: Peak GPU Memory per Epoch
    plt.figure(figsize=(8, 6))
    for rank in ranks_to_plot:
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
    plt.savefig(f"{OUTPUT_DIR}/peak_memory_per_epoch_across_ranks.png", dpi=300)
    plt.close()
    print("✓ Saved peak_memory_per_epoch_across_ranks.png")
    
    # Plot 3: FLOPs per Step vs Rank
    if flops_summary:
        plt.figure(figsize=(8, 5))
        ranks_sorted = sorted(flops_summary.keys())
        values = [flops_summary[r] / 1e12 for r in ranks_sorted]
        labels = [f"r{r}" for r in ranks_sorted]
        plt.bar(labels, values, color=plt.cm.viridis(range(len(ranks_sorted))))
        plt.xlabel("Rank")
        plt.ylabel("FLOPs per Step (TFLOPs)")
        plt.title("Compute Cost per Step vs rSVD Rank")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/flops_vs_rank.png", dpi=300)
        plt.close()
        print("✓ Saved flops_vs_rank.png")
    
    # Plot 3b: Epoch FLOPs vs Rank
    if epoch_flops_summary:
        plt.figure(figsize=(8, 5))
        ranks_sorted = sorted(epoch_flops_summary.keys())
        values = [epoch_flops_summary[r] / 1e15 for r in ranks_sorted]
        labels = [f"r{r}" for r in ranks_sorted]
        plt.bar(labels, values, color=plt.cm.plasma(range(len(ranks_sorted))))
        plt.xlabel("Rank")
        plt.ylabel("FLOPs per Epoch (PFLOPs)")
        plt.title("Compute Cost per Epoch vs rSVD Rank")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/epoch_flops_vs_rank.png", dpi=300)
        plt.close()
        print("✓ Saved epoch_flops_vs_rank.png")
    
    # Plot 4: Peak GPU Memory vs Rank
    if memory_summary:
        plt.figure(figsize=(8, 5))
        ranks_sorted = sorted(memory_summary.keys())
        values = [memory_summary[r] for r in ranks_sorted]
        labels = [f"r{r}" for r in ranks_sorted]
        plt.bar(labels, values, color=plt.cm.coolwarm(range(len(ranks_sorted))))
        plt.xlabel("Rank")
        plt.ylabel("Peak Memory (GB)")
        plt.title("Peak GPU Memory vs rSVD Rank")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/peak_memory_vs_rank.png", dpi=300)
        plt.close()
        print("✓ Saved peak_memory_vs_rank.png")
    
    # Plot 5: Total Program Memory vs Rank
    for rank in ranks_to_plot:
        if rank not in data:
            continue
        total_mem_row = data[rank]["total_mem"].loc[data[rank]["total_mem"]["metric"] == "program_total_peak_memory"]
        if not total_mem_row.empty:
            program_mem_summary[rank] = total_mem_row["value_bytes"].item() / 1e9
    
    if program_mem_summary:
        plt.figure(figsize=(8, 5))
        ranks_sorted = sorted(program_mem_summary.keys())
        plt.plot(ranks_sorted, [program_mem_summary[r] for r in ranks_sorted], marker="o", linewidth=2, markersize=8)
        plt.xlabel("Rank")
        plt.ylabel("Memory (GB)")
        plt.title("Total Program Peak Memory vs Rank")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/total_program_memory_vs_rank.png", dpi=300)
        plt.close()
        print("✓ Saved total_program_memory_vs_rank.png")
    
    # Plot 6: Forward Pass Memory vs Rank
    if forward_mem_summary:
        plt.figure(figsize=(8, 5))
        ranks_sorted = sorted(forward_mem_summary.keys())
        plt.plot(ranks_sorted, [forward_mem_summary[r] for r in ranks_sorted], marker="o", linewidth=2, markersize=8, color="green")
        plt.xlabel("Rank")
        plt.ylabel("Peak Forward Memory (GB)")
        plt.title("Forward Pass Peak Memory vs Rank")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/forward_memory_vs_rank.png", dpi=300)
        plt.close()
        print("✓ Saved forward_memory_vs_rank.png")
    
    # Plot 7: Forward Pass FLOPs vs Rank
    if forward_flops_summary:
        plt.figure(figsize=(8, 5))
        ranks_sorted = sorted(forward_flops_summary.keys())
        plt.plot(ranks_sorted, [forward_flops_summary[r] / 1e9 for r in ranks_sorted], marker="o", linewidth=2, markersize=8, color="purple")
        plt.xlabel("Rank")
        plt.ylabel("Forward FLOPs (GFLOPs)")
        plt.title("Forward Pass FLOPs vs Rank")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/forward_flops_vs_rank.png", dpi=300)
        plt.close()
        print("✓ Saved forward_flops_vs_rank.png")
    
    print(f"\n✓ All multi-rank plots saved to: {OUTPUT_DIR}/")


# =========================================
# MEMORY BREAKDOWN PLOTS
# =========================================
def load_memory_breakdowns(base_path: str = BASE) -> Dict[str, pd.DataFrame]:
    """Load all memory breakdown CSV files."""
    pattern = os.path.join(base_path, "memory_breakdown*.csv")
    files = glob.glob(pattern)
    
    data = {}
    for file in files:
        filename = os.path.basename(file)
        rank = parse_rank_from_filename(filename)
        
        try:
            df = pd.read_csv(file)
            if rank is None:
                key = "Adam"
            else:
                key = f"r{rank}"
            data[key] = df
            print(f"✓ Loaded {filename} -> {key}")
        except Exception as e:
            print(f"✗ Error loading {filename}: {e}")
    
    return data


def plot_memory_breakdown():
    """Generate memory breakdown plots."""
    print("\n" + "=" * 80)
    print("Generating Memory Breakdown Plots")
    print("=" * 80)
    
    # Use current directory if we're already in graph/, otherwise use BASE
    import os
    if os.path.basename(os.getcwd()) == "graph":
        base_path = "."
    else:
        base_path = BASE
    
    print("\nLoading memory breakdown data...")
    data = load_memory_breakdowns(base_path)
    
    if not data:
        print("No memory breakdown files found!")
        return
    
    print(f"\nFound {len(data)} memory breakdown files")
    
    # Generate all plots
    print("\nGenerating plots...")
    print("-" * 80)
    
    # Plot 1: Main components comparison
    ranks = []
    parameters = []
    optimizer_states = []
    activations = []
    gradients = []
    
    sorted_keys = sorted([k for k in data.keys() if k != "Adam"], 
                        key=lambda x: int(x[1:]) if x.startswith('r') else 999)
    sorted_keys = ["Adam"] + sorted_keys
    
    for key in sorted_keys:
        df = data[key]
        if key == "Adam":
            ranks.append("Adam")
        else:
            ranks.append(key.upper())
        parameters.append(extract_component_value(df, "parameters_total"))
        optimizer_states.append(extract_component_value(df, "optimizer_states_total"))
        activations.append(extract_component_value(df, "activations_total"))
        gradients.append(extract_component_value(df, "gradients_total"))
    
    x = range(len(ranks))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - 1.5*width for i in x], parameters, width, label='Parameters', color='#1f77b4')
    ax.bar([i - 0.5*width for i in x], optimizer_states, width, label='Optimizer States', color='#ff7f0e')
    ax.bar([i + 0.5*width for i in x], activations, width, label='Activations', color='#2ca02c')
    ax.bar([i + 1.5*width for i in x], gradients, width, label='Gradients', color='#d62728')
    ax.set_xlabel('Optimizer / Rank', fontsize=12)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('Memory Breakdown: Main Components Across Ranks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ranks)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "memory_components_comparison.png"), dpi=300)
    plt.close()
    print("✓ Saved memory_components_comparison.png")
    
    # Plot 2: Stacked memory composition
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, parameters, label='Parameters', color='#1f77b4')
    ax.bar(x, optimizer_states, bottom=parameters, label='Optimizer States', color='#ff7f0e')
    ax.bar(x, activations, bottom=[p+o for p, o in zip(parameters, optimizer_states)], 
           label='Activations', color='#2ca02c')
    ax.bar(x, gradients, bottom=[p+o+a for p, o, a in zip(parameters, optimizer_states, activations)], 
           label='Gradients', color='#d62728')
    ax.set_xlabel('Optimizer / Rank', fontsize=12)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('Stacked Memory Composition Across Ranks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ranks)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "memory_composition_stacked.png"), dpi=300)
    plt.close()
    print("✓ Saved memory_composition_stacked.png")
    
    # Plot 3: Optimizer state breakdown
    rsvd_keys = sorted([k for k in data.keys() if k.startswith('r')], 
                      key=lambda x: int(x[1:]))
    if rsvd_keys:
        state_types = set()
        rank_data = {}
        for key in rsvd_keys:
            df = data[key]
            rank_data[key] = {}
            opt_rows = df[df['component'].str.startswith('optimizer_state_')]
            for _, row in opt_rows.iterrows():
                state_name = row['component'].replace('optimizer_state_', '')
                if state_name != 'total':
                    state_types.add(state_name)
                    rank_data[key][state_name] = row['memory_mb']
        
        state_types = sorted(list(state_types))
        x_opt = range(len(rsvd_keys))
        width_opt = 0.8 / len(state_types)
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.Set3(range(len(state_types)))
        for i, state_type in enumerate(state_types):
            values = [rank_data[key].get(state_type, 0) for key in rsvd_keys]
            offset = (i - len(state_types)/2) * width_opt + width_opt/2
            ax.bar([xi + offset for xi in x_opt], values, width_opt, label=state_type, color=colors[i])
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('Memory (MB)', fontsize=12)
        ax.set_title('rSVD Optimizer State Breakdown by Rank', fontsize=14, fontweight='bold')
        ax.set_xticks(x_opt)
        ax.set_xticklabels([k.upper() for k in rsvd_keys])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "optimizer_state_breakdown.png"), dpi=300)
        plt.close()
        print("✓ Saved optimizer_state_breakdown.png")
    
    # Plot 4: Total memory comparison
    total_memory = []
    for key in sorted_keys:
        df = data[key]
        total = extract_component_value(df, "total_components")
        total_memory.append(total)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4' if r == "Adam" else '#ff7f0e' for r in ranks]
    bars = ax.bar(ranks, total_memory, color=colors, alpha=0.8)
    for bar, val in zip(bars, total_memory):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} MB',
                ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Optimizer / Rank', fontsize=12)
    ax.set_ylabel('Total Memory (MB)', fontsize=12)
    ax.set_title('Total Memory Usage: Parameters + Optimizer + Activations + Gradients', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "total_memory_comparison.png"), dpi=300)
    plt.close()
    print("✓ Saved total_memory_comparison.png")
    
    # Plot 5: Optimizer savings
    if "Adam" in data:
        adam_opt_memory = extract_component_value(data["Adam"], "optimizer_states_total")
        rsvd_keys = sorted([k for k in data.keys() if k.startswith('r')], 
                          key=lambda x: int(x[1:]))
        if rsvd_keys:
            opt_ranks = []
            opt_memory = []
            savings = []
            for key in rsvd_keys:
                opt_ranks.append(int(key[1:]))
                opt_mem = extract_component_value(data[key], "optimizer_states_total")
                opt_memory.append(opt_mem)
                savings.append(adam_opt_memory - opt_mem)
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            bars = ax1.bar([f"R{r}" for r in opt_ranks], opt_memory, color='#ff7f0e', alpha=0.7, label='rSVD Optimizer Memory')
            ax1.axhline(y=adam_opt_memory, color='#1f77b4', linestyle='--', linewidth=2, label='Adam Optimizer Memory')
            ax1.set_xlabel('Rank', fontsize=12)
            ax1.set_ylabel('Optimizer State Memory (MB)', fontsize=12, color='#ff7f0e')
            ax1.tick_params(axis='y', labelcolor='#ff7f0e')
            ax1.grid(True, alpha=0.3, axis='y')
            ax2 = ax1.twinx()
            ax2.plot([f"R{r}" for r in opt_ranks], savings, color='#2ca02c', marker='o', 
                     linewidth=2, markersize=8, label='Memory Savings')
            ax2.set_ylabel('Memory Savings vs Adam (MB)', fontsize=12, color='#2ca02c')
            ax2.tick_params(axis='y', labelcolor='#2ca02c')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            ax1.set_title('Optimizer State Memory: rSVD vs Adam', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "optimizer_memory_savings_breakdown.png"), dpi=300)
            plt.close()
            print("✓ Saved optimizer_memory_savings_breakdown.png")
    
    # Plot 6: CUDA memory breakdown
    allocated = []
    reserved = []
    peak_allocated = []
    for key in sorted_keys:
        df = data[key]
        allocated.append(extract_component_value(df, "cuda_allocated_mb"))
        reserved.append(extract_component_value(df, "cuda_reserved_mb"))
        peak_allocated.append(extract_component_value(df, "cuda_peak_allocated_mb"))
    
    x_cuda = range(len(ranks))
    width_cuda = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width_cuda for i in x_cuda], allocated, width_cuda, label='Allocated', color='#1f77b4')
    ax.bar(x_cuda, reserved, width_cuda, label='Reserved', color='#ff7f0e')
    ax.bar([i + width_cuda for i in x_cuda], peak_allocated, width_cuda, label='Peak Allocated', color='#2ca02c')
    ax.set_xlabel('Optimizer / Rank', fontsize=12)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('CUDA Memory Breakdown Across Ranks', fontsize=14, fontweight='bold')
    ax.set_xticks(x_cuda)
    ax.set_xticklabels(ranks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cuda_memory_breakdown.png"), dpi=300)
    plt.close()
    print("✓ Saved cuda_memory_breakdown.png")
    
    print("\n✓ All memory breakdown plots generated!")


# =========================================
# OPTIMIZER MEMORY PLOTS
# =========================================
def plot_optimizer_memory():
    """Generate optimizer memory comparison plots."""
    print("\n" + "=" * 80)
    print("Generating Optimizer Memory Plots")
    print("=" * 80)
    
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
            try:
                rank_str = dirname[1:-4]
                rank = int(rank_str)
                detected_ranks.append(rank)
            except ValueError:
                continue
    
    detected_ranks = sorted(detected_ranks)
    print(f"Detected ranks: {detected_ranks}")
    print(f"Adam directory: {adam_dir}")
    
    if not adam_dir or not os.path.exists(adam_dir):
        print("⚠ Warning: adam_opt directory not found!")
        return
    
    if not detected_ranks:
        print("⚠ Warning: No r*_opt directories found!")
        return
    
    # Load data
    adam_memory = None
    adam_csv = os.path.join(adam_dir, "optimizer_memory.csv")
    if os.path.exists(adam_csv):
        adam_df = pd.read_csv(adam_csv)
        adam_memory = adam_df.iloc[0]['memory_mb']
        print(f"✓ Loaded Adam optimizer memory: {adam_memory:.2f} MB")
    else:
        print(f"⚠ Error: {adam_csv} not found!")
        return
    
    rsvd_data = {}
    for rank in detected_ranks:
        rank_dir = os.path.join(BASE, f"r{rank}_opt")
        rsvd_csv = os.path.join(rank_dir, "optimizer_memory.csv")
        if os.path.exists(rsvd_csv):
            rsvd_df = pd.read_csv(rsvd_csv)
            rsvd_data[rank] = {
                'memory_mb': rsvd_df.iloc[0]['memory_mb'],
                'savings_mb': rsvd_df.iloc[0]['savings_mb'],
                'savings_pct': rsvd_df.iloc[0]['savings_pct']
            }
            print(f"✓ Loaded rSVD (rank={rank}) optimizer memory: {rsvd_data[rank]['memory_mb']:.2f} MB")
    
    if not rsvd_data:
        print("ERROR: No rSVD optimizer memory data found!")
        return
    
    # Plot 1: Optimizer Memory vs Rank
    plt.figure(figsize=(10, 6))
    ranks_sorted = sorted(rsvd_data.keys())
    values = [rsvd_data[r]['memory_mb'] for r in ranks_sorted]
    labels = [f"r{r}" for r in ranks_sorted]
    all_labels = ["Adam"] + labels
    all_values = [adam_memory] + values
    colors = ['#1f77b4'] + [plt.cm.viridis(i / len(ranks_sorted)) for i in range(len(ranks_sorted))]
    plt.bar(all_labels, all_values, color=colors)
    plt.xlabel("Optimizer / Rank", fontsize=12)
    plt.ylabel("Optimizer State Memory (MB)", fontsize=12)
    plt.title("Optimizer State Memory: Adam vs rSVD Adam", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/optimizer_memory_comparison.png", dpi=300)
    plt.close()
    print("✓ Saved optimizer_memory_comparison.png")
    
    # Plot 2: Memory Savings vs Rank
    savings_mb = [rsvd_data[r]['savings_mb'] for r in ranks_sorted]
    savings_pct = [rsvd_data[r]['savings_pct'] for r in ranks_sorted]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    bars = ax1.bar([f"r{r}" for r in ranks_sorted], savings_mb, color=plt.cm.plasma(range(len(ranks_sorted))), 
                    alpha=0.7, label='Savings (MB)')
    ax1.set_xlabel("Rank", fontsize=12)
    ax1.set_ylabel("Memory Savings (MB)", fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    line = ax2.plot([f"r{r}" for r in ranks_sorted], savings_pct, marker='o', linewidth=2, 
                    markersize=8, color='red', label='Savings (%)')
    ax2.set_ylabel("Memory Savings (%)", fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.title("Optimizer Memory Savings vs Rank", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/optimizer_memory_savings.png", dpi=300)
    plt.close()
    print("✓ Saved optimizer_memory_savings.png")
    
    # Plot 3: Optimizer Memory vs Rank (rSVD only)
    plt.figure(figsize=(8, 5))
    plt.plot(ranks_sorted, values, marker="o", linewidth=2, markersize=8, color="green")
    plt.xlabel("Rank", fontsize=12)
    plt.ylabel("Optimizer State Memory (MB)", fontsize=12)
    plt.title("rSVD Adam Optimizer Memory vs Rank", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/optimizer_memory_vs_rank.png", dpi=300)
    plt.close()
    print("✓ Saved optimizer_memory_vs_rank.png")
    
    # Plot 4: Side-by-side comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(ranks_sorted) + 1)
    width = 0.35
    adam_values = [adam_memory] * (len(ranks_sorted) + 1)
    rsvd_values = [adam_memory] + values
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
    plt.savefig(f"{OUTPUT_DIR}/optimizer_memory_side_by_side.png", dpi=300)
    plt.close()
    print("✓ Saved optimizer_memory_side_by_side.png")
    
    print(f"\n✓ All optimizer memory plots saved to: {OUTPUT_DIR}/")


# =========================================
# LORA vs RSVD COMPARISON PLOTS
# =========================================
def load_data(base_path, rank, method_name, rank_dir_template="r{rank}"):
    """Load data for a specific rank."""
    rank_dir = os.path.join(base_path, rank_dir_template.format(rank=rank))
    if not os.path.exists(rank_dir):
        return None
    
    data = {}
    try:
        loss_path = os.path.join(rank_dir, "epoch_loss.csv")
        if os.path.exists(loss_path):
            data['loss'] = pd.read_csv(loss_path)
            data['loss']['eval_loss'] = pd.to_numeric(data['loss']['eval_loss'], errors='coerce')
        
        mem_path = os.path.join(rank_dir, "epoch_peak_memory.csv")
        if os.path.exists(mem_path):
            data['peak_memory'] = pd.read_csv(mem_path)
        
        flops_path = os.path.join(rank_dir, "flops_profiler_stats.csv")
        if os.path.exists(flops_path):
            data['flops'] = pd.read_csv(flops_path)
        
        total_mem_path = os.path.join(rank_dir, "total_program_memory.csv")
        if os.path.exists(total_mem_path):
            data['total_memory'] = pd.read_csv(total_mem_path)
        
        forward_dir = os.path.join(rank_dir, "forward_pass")
        if os.path.exists(forward_dir):
            forward_mem_path = os.path.join(forward_dir, "peak_forward_memory.csv")
            if os.path.exists(forward_mem_path):
                data['forward_memory'] = pd.read_csv(forward_mem_path)
            forward_flops_path = os.path.join(forward_dir, "forward_flops.csv")
            if os.path.exists(forward_flops_path):
                data['forward_flops'] = pd.read_csv(forward_flops_path)
        
        return data
    except Exception as e:
        print(f"Error loading {method_name} rank {rank}: {e}")
        return None


def plot_lora_rsvd_comparison():
    """Generate LoRA vs rSVD comparison plots."""
    print("\n" + "=" * 80)
    print("Generating LoRA vs rSVD Comparison Plots")
    print("=" * 80)
    print(f"\nComparing ranks: {RANKS}")
    
    # Load data
    print("\nLoading data...")
    lora_data = {}
    rsvd_data = {}
    rsvd_weight_reduction_data = {}
    rsvd_svt_data = {}
    
    for rank in RANKS:
        lora_data[rank] = load_data(LORA_BASE_PATH, rank, "LoRA")
        rsvd_data[rank] = load_data(RSVD_BASE_PATH, rank, "rSVD")
        if os.path.exists(RSVD_WEIGHT_REDUCTION_BASE_PATH):
            rsvd_weight_reduction_data[rank] = load_data(
                RSVD_WEIGHT_REDUCTION_BASE_PATH, rank, "rSVD Weight Reduction"
            )
        else:
            rsvd_weight_reduction_data[rank] = None
        
        if os.path.exists(RSVD_SVT_BASE_PATH):
            rsvd_svt_data[rank] = load_data(RSVD_SVT_BASE_PATH, rank, "rSVD SVT")
        else:
            rsvd_svt_data[rank] = None
        
        if lora_data[rank]:
            print(f"  ✓ Loaded LoRA r{rank}")
        if rsvd_data[rank]:
            print(f"  ✓ Loaded rSVD r{rank}")
        if rsvd_weight_reduction_data[rank]:
            print(f"  ✓ Loaded rSVD Weight Reduction r{rank}")
        if rsvd_svt_data[rank]:
            print(f"  ✓ Loaded rSVD SVT r{rank}")
    
    print("\nGenerating comparison plots...")
    print("-" * 80)
    
    # Plot 1: Loss comparison across ranks
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    for rank in RANKS:
        if rank in lora_data and lora_data[rank] and 'loss' in lora_data[rank]:
            df = lora_data[rank]['loss']
            ax1.plot(df['epoch'], df['train_loss'], marker='o', 
                    label=f'LoRA r{rank}', linewidth=2, alpha=0.8)
        if rank in rsvd_data and rsvd_data[rank] and 'loss' in rsvd_data[rank]:
            df = rsvd_data[rank]['loss']
            ax1.plot(df['epoch'], df['train_loss'], marker='s', 
                    label=f'rSVD r{rank}', linewidth=2, alpha=0.8, linestyle='--')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss: LoRA vs rSVD Across Ranks', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_comparison_across_ranks.png"), dpi=300)
    plt.close()
    print("✓ Saved loss_comparison_across_ranks.png")
    
    # Plot 2: Loss comparison with weight reduction
    if any(rsvd_weight_reduction_data.values()):
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        for rank in RANKS:
            if rank in lora_data and lora_data[rank] and 'loss' in lora_data[rank]:
                df = lora_data[rank]['loss']
                ax1.plot(df['epoch'], df['train_loss'], marker='o',
                        label=f'LoRA r{rank}', linewidth=2, alpha=0.8)
            if (rank in rsvd_weight_reduction_data and rsvd_weight_reduction_data[rank] 
                and 'loss' in rsvd_weight_reduction_data[rank]):
                df = rsvd_weight_reduction_data[rank]['loss']
                ax1.plot(df['epoch'], df['train_loss'], marker='D',
                        label=f'rSVD WR r{rank}', linewidth=2, alpha=0.8, linestyle='--')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title('Training Loss: LoRA vs rSVD Weight Reduction Across Ranks', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "loss_comparison_across_ranks_weight_reduction.png"), dpi=300)
        plt.close()
        print("✓ Saved loss_comparison_across_ranks_weight_reduction.png")
    
    # Plot 3: Final loss vs rank
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    lora_train_losses = []
    rsvd_train_losses = []
    lora_eval_losses = []
    rsvd_eval_losses = []
    available_ranks = []
    
    for rank in RANKS:
        lora_train = None
        lora_eval = None
        rsvd_train = None
        rsvd_eval = None
        
        if rank in lora_data and lora_data[rank] and 'loss' in lora_data[rank]:
            df = lora_data[rank]['loss']
            lora_train = df['train_loss'].iloc[-1]
            if df['eval_loss'].notna().any():
                lora_eval = df['eval_loss'].dropna().iloc[-1]
        
        if rank in rsvd_data and rsvd_data[rank] and 'loss' in rsvd_data[rank]:
            df = rsvd_data[rank]['loss']
            rsvd_train = df['train_loss'].iloc[-1]
            if df['eval_loss'].notna().any():
                rsvd_eval = df['eval_loss'].dropna().iloc[-1]
        
        if lora_train is not None or rsvd_train is not None:
            available_ranks.append(rank)
            lora_train_losses.append(lora_train if lora_train is not None else np.nan)
            rsvd_train_losses.append(rsvd_train if rsvd_train is not None else np.nan)
            lora_eval_losses.append(lora_eval if lora_eval is not None else np.nan)
            rsvd_eval_losses.append(rsvd_eval if rsvd_eval is not None else np.nan)
    
    if available_ranks:
        ax1.plot(available_ranks, lora_train_losses, marker='o', label='LoRA', 
                linewidth=2, markersize=8, color='#ff7f0e')
        ax1.plot(available_ranks, rsvd_train_losses, marker='s', label='rSVD', 
                linewidth=2, markersize=8, color='#2ca02c')
        ax1.set_xlabel('Rank', fontsize=12)
        ax1.set_ylabel('Final Training Loss', fontsize=12)
        ax1.set_title('Final Training Loss vs Rank', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        ax1.set_xticks(available_ranks)
        
        ax2.plot(available_ranks, lora_eval_losses, marker='o', label='LoRA', 
                linewidth=2, markersize=8, color='#ff7f0e')
        ax2.plot(available_ranks, rsvd_eval_losses, marker='s', label='rSVD', 
                linewidth=2, markersize=8, color='#2ca02c')
        ax2.set_xlabel('Rank', fontsize=12)
        ax2.set_ylabel('Final Validation Loss', fontsize=12)
        ax2.set_title('Final Validation Loss vs Rank', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        ax2.set_xticks(available_ranks)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_loss_vs_rank.png"), dpi=300)
    plt.close()
    print("✓ Saved final_loss_vs_rank.png")
    
    # Plot 4: Memory vs rank (bar graphs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    lora_peak_mem = []
    rsvd_peak_mem = []
    rsvd_wr_peak_mem = []
    lora_total_mem = []
    rsvd_total_mem = []
    rsvd_wr_total_mem = []
    available_ranks_mem = []
    
    for rank in RANKS:
        lora_peak = None
        rsvd_peak = None
        rsvd_wr_peak = None
        lora_total = None
        rsvd_total = None
        rsvd_wr_total = None
        
        if rank in lora_data and lora_data[rank]:
            if 'peak_memory' in lora_data[rank]:
                lora_peak = lora_data[rank]['peak_memory']['peak_memory_bytes'].max() / 1e9
            if 'total_memory' in lora_data[rank]:
                df = lora_data[rank]['total_memory']
                total = df.loc[df['metric'] == 'program_total_peak_memory', 'value_bytes'].values
                if len(total) > 0:
                    lora_total = total[0] / 1e9
        
        if rank in rsvd_data and rsvd_data[rank]:
            if 'peak_memory' in rsvd_data[rank]:
                rsvd_peak = rsvd_data[rank]['peak_memory']['peak_memory_bytes'].max() / 1e9
            if 'total_memory' in rsvd_data[rank]:
                df = rsvd_data[rank]['total_memory']
                total = df.loc[df['metric'] == 'program_total_peak_memory', 'value_bytes'].values
                if len(total) > 0:
                    rsvd_total = total[0] / 1e9
        
        if rsvd_weight_reduction_data and rank in rsvd_weight_reduction_data and rsvd_weight_reduction_data[rank]:
            if 'peak_memory' in rsvd_weight_reduction_data[rank]:
                rsvd_wr_peak = rsvd_weight_reduction_data[rank]['peak_memory']['peak_memory_bytes'].max() / 1e9
            if 'total_memory' in rsvd_weight_reduction_data[rank]:
                df = rsvd_weight_reduction_data[rank]['total_memory']
                total = df.loc[df['metric'] == 'program_total_peak_memory', 'value_bytes'].values
                if len(total) > 0:
                    rsvd_wr_total = total[0] / 1e9
        
        if lora_peak is not None or rsvd_peak is not None or rsvd_wr_peak is not None:
            available_ranks_mem.append(rank)
            lora_peak_mem.append(lora_peak if lora_peak is not None else np.nan)
            rsvd_peak_mem.append(rsvd_peak if rsvd_peak is not None else np.nan)
            rsvd_wr_peak_mem.append(rsvd_wr_peak if rsvd_wr_peak is not None else np.nan)
            lora_total_mem.append(lora_total if lora_total is not None else np.nan)
            rsvd_total_mem.append(rsvd_total if rsvd_total is not None else np.nan)
            rsvd_wr_total_mem.append(rsvd_wr_total if rsvd_wr_total is not None else np.nan)
    
    if available_ranks_mem:
        x = np.arange(len(available_ranks_mem))
        width = 0.25
        bars1 = ax1.bar(x - width, lora_peak_mem, width, label='LoRA',
                        color='#ff7f0e', alpha=0.8)
        bars2 = ax1.bar(x, rsvd_peak_mem, width, label='rSVD Optimizer',
                        color='#2ca02c', alpha=0.8)
        bars3 = ax1.bar(x + width, rsvd_wr_peak_mem, width, label='rSVD Weight Reduction',
                        color='#1f77b4', alpha=0.8)
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        ax1.set_xlabel('Rank', fontsize=12)
        ax1.set_ylabel('Peak Memory (GB)', fontsize=12)
        ax1.set_title('Peak Training Memory vs Rank', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'r{r}' for r in available_ranks_mem])
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        bars1 = ax2.bar(x - width, lora_total_mem, width, label='LoRA',
                        color='#ff7f0e', alpha=0.8)
        bars2 = ax2.bar(x, rsvd_total_mem, width, label='rSVD Optimizer',
                        color='#2ca02c', alpha=0.8)
        bars3 = ax2.bar(x + width, rsvd_wr_total_mem, width, label='rSVD Weight Reduction',
                        color='#1f77b4', alpha=0.8)
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        ax2.set_xlabel('Rank', fontsize=12)
        ax2.set_ylabel('Total Program Memory (GB)', fontsize=12)
        ax2.set_title('Total Program Memory vs Rank', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'r{r}' for r in available_ranks_mem])
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "memory_vs_rank.png"), dpi=300)
    plt.close()
    print("✓ Saved memory_vs_rank.png")
    
    # Plot 5: FLOPs vs rank
    fig, ax = plt.subplots(figsize=(10, 6))
    lora_flops = []
    rsvd_flops = []
    rsvd_svt_flops = []
    available_ranks_flops = []
    
    for rank in RANKS:
        lora_flop = None
        rsvd_flop = None
        rsvd_svt_flop = None
        
        if rank in lora_data and lora_data[rank] and 'flops' in lora_data[rank]:
            df = lora_data[rank]['flops']
            step_flops = df.loc[df['metric'] == 'step_flops', 'value'].values
            if len(step_flops) > 0:
                lora_flop = step_flops[0] / 1e12
        
        if rank in rsvd_data and rsvd_data[rank] and 'flops' in rsvd_data[rank]:
            df = rsvd_data[rank]['flops']
            step_flops = df.loc[df['metric'] == 'step_flops', 'value'].values
            if len(step_flops) > 0:
                rsvd_flop = step_flops[0] / 1e12
        
        if rsvd_svt_data and rank in rsvd_svt_data and rsvd_svt_data[rank] and 'flops' in rsvd_svt_data[rank]:
            df = rsvd_svt_data[rank]['flops']
            step_flops = df.loc[df['metric'] == 'step_flops', 'value'].values
            if len(step_flops) > 0:
                rsvd_svt_flop = step_flops[0] / 1e12
        
        if lora_flop is not None or rsvd_flop is not None or rsvd_svt_flop is not None:
            available_ranks_flops.append(rank)
            lora_flops.append(lora_flop if lora_flop is not None else np.nan)
            rsvd_flops.append(rsvd_flop if rsvd_flop is not None else np.nan)
            rsvd_svt_flops.append(rsvd_svt_flop if rsvd_svt_flop is not None else np.nan)
    
    if available_ranks_flops:
        ax.plot(available_ranks_flops, lora_flops, marker='o', label='LoRA', 
                linewidth=2, markersize=8, color='#ff7f0e')
        ax.plot(available_ranks_flops, rsvd_flops, marker='s', label='rSVD', 
                linewidth=2, markersize=8, color='#2ca02c')
        if rsvd_svt_data:
            ax.plot(available_ranks_flops, rsvd_svt_flops, marker='^', label='rSVD SVT', 
                    linewidth=2, markersize=8, color='#9467bd', linestyle=':')
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('FLOPs per Step (TFLOPs)', fontsize=12)
        ax.set_title('Compute Cost (FLOPs per Step) vs Rank', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xticks(available_ranks_flops)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "flops_comparison_lora_rsvd.png"), dpi=300)
    plt.close()
    print("✓ Saved flops_comparison_lora_rsvd.png")
    
    # Plot 6: Summary by rank
    ranks_list = []
    for rank in RANKS:
        if (rank in lora_data and lora_data[rank]) or (rank in rsvd_data and rsvd_data[rank]):
            ranks_list.append(rank)
    
    if len(ranks_list) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Final Training Loss by Rank
        ax = axes[0, 0]
        lora_train = []
        rsvd_train = []
        for rank in ranks_list:
            if rank in lora_data and lora_data[rank] and 'loss' in lora_data[rank]:
                final_loss = lora_data[rank]['loss']['train_loss'].iloc[-1]
                lora_train.append(final_loss)
            else:
                lora_train.append(np.nan)
            if rank in rsvd_data and rsvd_data[rank] and 'loss' in rsvd_data[rank]:
                final_loss = rsvd_data[rank]['loss']['train_loss'].iloc[-1]
                rsvd_train.append(final_loss)
            else:
                rsvd_train.append(np.nan)
        
        x = np.arange(len(ranks_list))
        width = 0.35
        ax.bar(x - width/2, lora_train, width, label='LoRA', color='#ff7f0e', alpha=0.8)
        ax.bar(x + width/2, rsvd_train, width, label='rSVD', color='#2ca02c', alpha=0.8)
        ax.set_xlabel('Rank', fontsize=11)
        ax.set_ylabel('Final Training Loss', fontsize=11)
        ax.set_title('Final Training Loss by Rank', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'r{r}' for r in ranks_list])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Final Validation Loss by Rank
        ax = axes[0, 1]
        lora_eval = []
        rsvd_eval = []
        for rank in ranks_list:
            if rank in lora_data and lora_data[rank] and 'loss' in lora_data[rank]:
                df = lora_data[rank]['loss']
                if df['eval_loss'].notna().any():
                    lora_eval.append(df['eval_loss'].dropna().iloc[-1])
                else:
                    lora_eval.append(np.nan)
            else:
                lora_eval.append(np.nan)
            if rank in rsvd_data and rsvd_data[rank] and 'loss' in rsvd_data[rank]:
                df = rsvd_data[rank]['loss']
                if df['eval_loss'].notna().any():
                    rsvd_eval.append(df['eval_loss'].dropna().iloc[-1])
                else:
                    rsvd_eval.append(np.nan)
            else:
                rsvd_eval.append(np.nan)
        
        ax.bar(x - width/2, lora_eval, width, label='LoRA', color='#ff7f0e', alpha=0.8)
        ax.bar(x + width/2, rsvd_eval, width, label='rSVD', color='#2ca02c', alpha=0.8)
        ax.set_xlabel('Rank', fontsize=11)
        ax.set_ylabel('Final Validation Loss', fontsize=11)
        ax.set_title('Final Validation Loss by Rank', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'r{r}' for r in ranks_list])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Peak Memory by Rank
        ax = axes[1, 0]
        lora_mem = []
        rsvd_mem = []
        for rank in ranks_list:
            if rank in lora_data and lora_data[rank] and 'peak_memory' in lora_data[rank]:
                lora_mem.append(lora_data[rank]['peak_memory']['peak_memory_bytes'].max() / 1e9)
            else:
                lora_mem.append(np.nan)
            if rank in rsvd_data and rsvd_data[rank] and 'peak_memory' in rsvd_data[rank]:
                rsvd_mem.append(rsvd_data[rank]['peak_memory']['peak_memory_bytes'].max() / 1e9)
            else:
                rsvd_mem.append(np.nan)
        
        ax.bar(x - width/2, lora_mem, width, label='LoRA', color='#ff7f0e', alpha=0.8)
        ax.bar(x + width/2, rsvd_mem, width, label='rSVD', color='#2ca02c', alpha=0.8)
        ax.set_xlabel('Rank', fontsize=11)
        ax.set_ylabel('Peak Memory (GB)', fontsize=11)
        ax.set_title('Peak Training Memory by Rank', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'r{r}' for r in ranks_list])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # FLOPs by Rank
        ax = axes[1, 1]
        lora_flops = []
        rsvd_flops = []
        for rank in ranks_list:
            if rank in lora_data and lora_data[rank] and 'flops' in lora_data[rank]:
                df = lora_data[rank]['flops']
                step_flops = df.loc[df['metric'] == 'step_flops', 'value'].values
                if len(step_flops) > 0:
                    lora_flops.append(step_flops[0] / 1e12)
                else:
                    lora_flops.append(np.nan)
            else:
                lora_flops.append(np.nan)
            if rank in rsvd_data and rsvd_data[rank] and 'flops' in rsvd_data[rank]:
                df = rsvd_data[rank]['flops']
                step_flops = df.loc[df['metric'] == 'step_flops', 'value'].values
                if len(step_flops) > 0:
                    rsvd_flops.append(step_flops[0] / 1e12)
                else:
                    rsvd_flops.append(np.nan)
            else:
                rsvd_flops.append(np.nan)
        
        ax.bar(x - width/2, lora_flops, width, label='LoRA', color='#ff7f0e', alpha=0.8)
        ax.bar(x + width/2, rsvd_flops, width, label='rSVD', color='#2ca02c', alpha=0.8)
        ax.set_xlabel('Rank', fontsize=11)
        ax.set_ylabel('FLOPs per Step (TFLOPs)', fontsize=11)
        ax.set_title('Compute Cost by Rank', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'r{r}' for r in ranks_list])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('LoRA vs rSVD: Summary Comparison Across Ranks', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "summary_comparison_across_ranks.png"), dpi=300)
        plt.close()
        print("✓ Saved summary_comparison_across_ranks.png")
    
    print("\n✓ All LoRA vs rSVD comparison plots generated!")


# =========================================
# ADAM vs RSVDADAM COMPARISON PLOT
# =========================================
def plot_adam_vs_rsvdadam_comparison(adam_losses, rsvd_losses, adam_acc=None, rsvd_acc=None, 
                                     output_path=None, save=True, show=False):
    """
    Plot training loss comparison between Adam and rSVDAdam optimizers.
    
    This function is used by training scripts to plot comparison results.
    Can be called with loss arrays directly (for use in training scripts).
    
    Args:
        adam_losses: List or array of training losses for Adam optimizer
        rsvd_losses: List or array of training losses for rSVDAdam optimizer
        adam_acc: Optional accuracy for Adam (to include in label)
        rsvd_acc: Optional accuracy for rSVDAdam (to include in label)
        output_path: Path to save plot (default: OUTPUT_DIR/loss_comparison_adam_rsvdadam.png)
        save: Whether to save the plot (default: True)
        show: Whether to show the plot (default: False)
    
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Create labels with or without accuracy
    adam_label = f"Adam ({adam_acc*100:.2f}% acc)" if adam_acc is not None else "Adam"
    rsvd_label = f"rSVDAdam ({rsvd_acc*100:.2f}% acc)" if rsvd_acc is not None else "rSVDAdam"
    
    ax.plot(adam_losses, label=adam_label, marker="o")
    ax.plot(rsvd_losses, label=rsvd_label, marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss: Adam vs rSVDAdam (with rSVD)")
    ax.legend()
    ax.grid(True)
    
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "loss_comparison_adam_rsvdadam.png")
    
    plt.tight_layout()
    
    if save:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved comparison plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =========================================
# MAIN FUNCTION
# =========================================
def main():
    """Main function to generate all plots."""
    print("=" * 80)
    print("rSVD Plotting Script")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Base directory: {BASE}")
    
    # Generate all plots
    plot_multi_rank_plots()
    plot_memory_breakdown()
    plot_optimizer_memory()
    plot_lora_rsvd_comparison()
    
    print("\n" + "=" * 80)
    print("All plots generated successfully!")
    print("=" * 80)
    print(f"\nPlots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

