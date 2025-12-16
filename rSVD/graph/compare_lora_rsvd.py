"""
Compare LoRA and rSVD at comparable ranks.

This script creates comparison plots between LoRA and rSVD across multiple ranks:
- r4, r8, r16, r64, r128

Metrics compared:
- Training and validation loss curves (across ranks)
- Peak memory per epoch (across ranks)
- Total program memory vs rank
- FLOPs per step vs rank
- Forward pass memory vs rank
- Summary comparison at each rank
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np


# Configuration
# Paths relative to where script is run (should be from rSVD/graph directory)
LORA_BASE_PATH = "../../Lora/graph/sst2"  # LoRA data is in sst2 subdirectory
RSVD_BASE_PATH = "."
RSVD_SVT_BASE_PATH = "../../rSVD_SVT_Fixed_Rank/sst/graph"  # rSVD SVT data
OUTPUT_DIR = "./plots"

# Define which ranks to compare (should match between LoRA and rSVD)
RANKS = [4, 8, 16, 64, 128]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(base_path, rank, method_name):
    """
    Load data for a specific rank.
    
    Args:
        base_path: Base path to the graph directory
        rank: Rank number
        method_name: Name of the method (for error messages)
    
    Returns:
        Dictionary with loaded dataframes, or None if data not found
    """
    rank_dir = os.path.join(base_path, f"r{rank}")
    
    if not os.path.exists(rank_dir):
        return None
    
    data = {}
    
    try:
        # Load epoch loss
        loss_path = os.path.join(rank_dir, "epoch_loss.csv")
        if os.path.exists(loss_path):
            data['loss'] = pd.read_csv(loss_path)
            data['loss']['eval_loss'] = pd.to_numeric(data['loss']['eval_loss'], errors='coerce')
        
        # Load peak memory per epoch
        mem_path = os.path.join(rank_dir, "epoch_peak_memory.csv")
        if os.path.exists(mem_path):
            data['peak_memory'] = pd.read_csv(mem_path)
        
        # Load FLOPs
        flops_path = os.path.join(rank_dir, "flops_profiler_stats.csv")
        if os.path.exists(flops_path):
            data['flops'] = pd.read_csv(flops_path)
        
        # Load total program memory
        total_mem_path = os.path.join(rank_dir, "total_program_memory.csv")
        if os.path.exists(total_mem_path):
            data['total_memory'] = pd.read_csv(total_mem_path)
        
        # Load forward pass data if available
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


def plot_loss_comparison_across_ranks(lora_data, rsvd_data):
    """Plot training and validation loss comparison across ranks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training Loss
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
    
    # Validation Loss
    for rank in RANKS:
        if rank in lora_data and lora_data[rank] and 'loss' in lora_data[rank]:
            df = lora_data[rank]['loss']
            if df['eval_loss'].notna().any():
                ax2.plot(df['epoch'], df['eval_loss'], marker='o', 
                        label=f'LoRA r{rank}', linewidth=2, alpha=0.8)
        
        if rank in rsvd_data and rsvd_data[rank] and 'loss' in rsvd_data[rank]:
            df = rsvd_data[rank]['loss']
            if df['eval_loss'].notna().any():
                ax2.plot(df['epoch'], df['eval_loss'], marker='s', 
                        label=f'rSVD r{rank}', linewidth=2, alpha=0.8, linestyle='--')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss: LoRA vs rSVD Across Ranks', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_comparison_across_ranks.png"), dpi=300)
    plt.close()
    print("✓ Saved loss_comparison_across_ranks.png")


def plot_final_loss_vs_rank(lora_data, rsvd_data):
    """Plot final training and validation loss vs rank."""
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
    
    # Training Loss
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
    
    # Validation Loss
    if available_ranks:
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


def plot_memory_vs_rank(lora_data, rsvd_data):
    """Plot peak memory and total memory vs rank as bar graphs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    lora_peak_mem = []
    rsvd_peak_mem = []
    lora_total_mem = []
    rsvd_total_mem = []
    available_ranks = []
    
    for rank in RANKS:
        lora_peak = None
        rsvd_peak = None
        lora_total = None
        rsvd_total = None
        
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
        
        if lora_peak is not None or rsvd_peak is not None:
            available_ranks.append(rank)
            lora_peak_mem.append(lora_peak if lora_peak is not None else np.nan)
            rsvd_peak_mem.append(rsvd_peak if rsvd_peak is not None else np.nan)
            lora_total_mem.append(lora_total if lora_total is not None else np.nan)
            rsvd_total_mem.append(rsvd_total if rsvd_total is not None else np.nan)
    
    # Peak Memory - Bar Chart
    if available_ranks:
        x = np.arange(len(available_ranks))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, lora_peak_mem, width, label='LoRA', 
                        color='#ff7f0e', alpha=0.8)
        bars2 = ax1.bar(x + width/2, rsvd_peak_mem, width, label='rSVD', 
                        color='#2ca02c', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Rank', fontsize=12)
        ax1.set_ylabel('Peak Memory (GB)', fontsize=12)
        ax1.set_title('Peak Training Memory vs Rank', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'r{r}' for r in available_ranks])
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Total Memory - Bar Chart
    if available_ranks:
        x = np.arange(len(available_ranks))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, lora_total_mem, width, label='LoRA', 
                        color='#ff7f0e', alpha=0.8)
        bars2 = ax2.bar(x + width/2, rsvd_total_mem, width, label='rSVD', 
                        color='#2ca02c', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax2.set_xlabel('Rank', fontsize=12)
        ax2.set_ylabel('Total Program Memory (GB)', fontsize=12)
        ax2.set_title('Total Program Memory vs Rank', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'r{r}' for r in available_ranks])
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "memory_vs_rank.png"), dpi=300)
    plt.close()
    print("✓ Saved memory_vs_rank.png")


def plot_flops_vs_rank(lora_data, rsvd_data, rsvd_svt_data=None):
    """Plot FLOPs per step vs rank."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lora_flops = []
    rsvd_flops = []
    rsvd_svt_flops = []
    available_ranks = []
    
    for rank in RANKS:
        lora_flop = None
        rsvd_flop = None
        rsvd_svt_flop = None
        
        if rank in lora_data and lora_data[rank] and 'flops' in lora_data[rank]:
            df = lora_data[rank]['flops']
            step_flops = df.loc[df['metric'] == 'step_flops', 'value'].values
            if len(step_flops) > 0:
                lora_flop = step_flops[0] / 1e12  # Convert to TFLOPs
        
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
            available_ranks.append(rank)
            lora_flops.append(lora_flop if lora_flop is not None else np.nan)
            rsvd_flops.append(rsvd_flop if rsvd_flop is not None else np.nan)
            rsvd_svt_flops.append(rsvd_svt_flop if rsvd_svt_flop is not None else np.nan)
    
    if available_ranks:
        ax.plot(available_ranks, lora_flops, marker='o', label='LoRA', 
                linewidth=2, markersize=8, color='#ff7f0e')
        ax.plot(available_ranks, rsvd_flops, marker='s', label='rSVD', 
                linewidth=2, markersize=8, color='#2ca02c')
        if rsvd_svt_data:
            ax.plot(available_ranks, rsvd_svt_flops, marker='^', label='rSVD SVT', 
                    linewidth=2, markersize=8, color='#9467bd', linestyle=':')
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('FLOPs per Step (TFLOPs)', fontsize=12)
        ax.set_title('Compute Cost (FLOPs per Step) vs Rank', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xticks(available_ranks)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "flops_comparison_lora_rsvd.png"), dpi=300)
    plt.close()
    print("✓ Saved flops_comparison_lora_rsvd.png")


def plot_summary_by_rank(lora_data, rsvd_data):
    """Create summary comparison plots grouped by rank."""
    # Build list of available ranks (where at least one method has data)
    ranks_list = []
    for rank in RANKS:
        if (rank in lora_data and lora_data[rank]) or (rank in rsvd_data and rsvd_data[rank]):
            ranks_list.append(rank)
    
    if len(ranks_list) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Final Training Loss by Rank
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
    
    # 2. Final Validation Loss by Rank
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
    
    # 3. Peak Memory by Rank
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
    
    # 4. FLOPs by Rank
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


def main():
    """Main function to generate all comparison plots."""
    print("=" * 80)
    print("LoRA vs rSVD Comparison Across Ranks")
    print("=" * 80)
    print(f"\nComparing ranks: {RANKS}")
    print()
    
    # Load data for all ranks
    print("Loading data...")
    lora_data = {}
    rsvd_data = {}
    rsvd_svt_data = {}
    
    for rank in RANKS:
        lora_data[rank] = load_data(LORA_BASE_PATH, rank, "LoRA")
        rsvd_data[rank] = load_data(RSVD_BASE_PATH, rank, "rSVD")
        
        # Try to load rSVD SVT data if path exists
        if os.path.exists(RSVD_SVT_BASE_PATH):
            rsvd_svt_data[rank] = load_data(RSVD_SVT_BASE_PATH, rank, "rSVD SVT")
        else:
            rsvd_svt_data[rank] = None
        
        if lora_data[rank]:
            print(f"  ✓ Loaded LoRA r{rank}")
        if rsvd_data[rank]:
            print(f"  ✓ Loaded rSVD r{rank}")
        if rsvd_svt_data[rank]:
            print(f"  ✓ Loaded rSVD SVT r{rank}")
    
    print("\nGenerating comparison plots...")
    print("-" * 80)
    
    # Generate all plots
    plot_loss_comparison_across_ranks(lora_data, rsvd_data)
    plot_final_loss_vs_rank(lora_data, rsvd_data)
    plot_memory_vs_rank(lora_data, rsvd_data)
    plot_flops_vs_rank(lora_data, rsvd_data, rsvd_svt_data if any(rsvd_svt_data.values()) else None)
    plot_summary_by_rank(lora_data, rsvd_data)
    
    print("\n" + "=" * 80)
    print("All comparison plots generated successfully!")
    print("=" * 80)
    print(f"\nPlots saved to: {OUTPUT_DIR}/")
    print("  - loss_comparison_across_ranks.png")
    print("  - final_loss_vs_rank.png")
    print("  - memory_vs_rank.png")
    print("  - flops_comparison_lora_rsvd.png")
    print("  - summary_comparison_across_ranks.png")


if __name__ == "__main__":
    main()
