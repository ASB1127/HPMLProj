"""
Plot memory breakdown comparisons from memory_breakdown CSV files.

This script creates visualizations comparing:
1. Main memory components (Parameters, Optimizer, Activations, Gradients) across ranks
2. Optimizer state memory breakdown
3. Total memory comparison (Adam vs rSVD at different ranks)
4. Stacked bar charts showing memory composition
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
from typing import Dict, Optional


def parse_rank_from_filename(filename: str) -> Optional[int]:
    """Extract rank from filename like 'memory_breakdown_r4.csv' or 'memory_breakdown.csv'."""
    match = re.search(r'r(\d+)', filename)
    if match:
        return int(match.group(1))
    return None  # Baseline (Adam)


def load_memory_breakdowns(base_path: str = "./graph") -> Dict[str, pd.DataFrame]:
    """
    Load all memory breakdown CSV files.
    
    Returns:
        Dictionary mapping rank identifier (or 'Adam' for baseline) to DataFrame
    """
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


def extract_component_value(df: pd.DataFrame, component_name: str) -> float:
    """Extract memory value for a specific component."""
    row = df[df['component'] == component_name]
    if len(row) > 0:
        return row.iloc[0]['memory_mb']
    return 0.0


def plot_main_components_comparison(data: Dict[str, pd.DataFrame], output_dir: str = "./plots"):
    """
    Plot main memory components (Parameters, Optimizer, Activations, Gradients) across ranks.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    ranks = []
    parameters = []
    optimizer_states = []
    activations = []
    gradients = []
    
    # Sort keys: Adam first, then by rank
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
    
    # Create grouped bar chart
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
    plt.savefig(os.path.join(output_dir, "memory_components_comparison.png"), dpi=300)
    plt.close()
    print("✓ Saved memory_components_comparison.png")


def plot_stacked_memory_composition(data: Dict[str, pd.DataFrame], output_dir: str = "./plots"):
    """
    Plot stacked bar chart showing memory composition.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    ranks = []
    parameters = []
    optimizer_states = []
    activations = []
    gradients = []
    
    # Sort keys: Adam first, then by rank
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
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(ranks))
    
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
    plt.savefig(os.path.join(output_dir, "memory_composition_stacked.png"), dpi=300)
    plt.close()
    print("✓ Saved memory_composition_stacked.png")


def plot_optimizer_state_breakdown(data: Dict[str, pd.DataFrame], output_dir: str = "./plots"):
    """
    Plot optimizer state memory breakdown for rSVD ranks.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Only plot rSVD ranks (not Adam)
    rsvd_keys = sorted([k for k in data.keys() if k.startswith('r')], 
                      key=lambda x: int(x[1:]))
    
    if not rsvd_keys:
        print("No rSVD data found for optimizer state breakdown")
        return
    
    # Collect optimizer state breakdowns
    state_types = set()
    rank_data = {}
    
    for key in rsvd_keys:
        df = data[key]
        rank_data[key] = {}
        
        # Get all optimizer state components
        opt_rows = df[df['component'].str.startswith('optimizer_state_')]
        for _, row in opt_rows.iterrows():
            state_name = row['component'].replace('optimizer_state_', '')
            if state_name != 'total':  # Skip the total row
                state_types.add(state_name)
                rank_data[key][state_name] = row['memory_mb']
    
    state_types = sorted(list(state_types))
    
    # Create grouped bar chart
    x = range(len(rsvd_keys))
    width = 0.8 / len(state_types)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.Set3(range(len(state_types)))
    
    for i, state_type in enumerate(state_types):
        values = [rank_data[key].get(state_type, 0) for key in rsvd_keys]
        offset = (i - len(state_types)/2) * width + width/2
        ax.bar([xi + offset for xi in x], values, width, label=state_type, color=colors[i])
    
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('rSVD Optimizer State Breakdown by Rank', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in rsvd_keys])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimizer_state_breakdown.png"), dpi=300)
    plt.close()
    print("✓ Saved optimizer_state_breakdown.png")


def plot_total_memory_comparison(data: Dict[str, pd.DataFrame], output_dir: str = "./plots"):
    """
    Plot total memory comparison (total_components) across ranks.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    ranks = []
    total_memory = []
    
    # Sort keys: Adam first, then by rank
    sorted_keys = sorted([k for k in data.keys() if k != "Adam"], 
                        key=lambda x: int(x[1:]) if x.startswith('r') else 999)
    sorted_keys = ["Adam"] + sorted_keys
    
    for key in sorted_keys:
        df = data[key]
        
        if key == "Adam":
            ranks.append("Adam")
        else:
            ranks.append(key.upper())
        
        total = extract_component_value(df, "total_components")
        total_memory.append(total)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4' if r == "Adam" else '#ff7f0e' for r in ranks]
    
    bars = ax.bar(ranks, total_memory, color=colors, alpha=0.8)
    
    # Add value labels on bars
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
    plt.savefig(os.path.join(output_dir, "total_memory_comparison.png"), dpi=300)
    plt.close()
    print("✓ Saved total_memory_comparison.png")


def plot_optimizer_savings(data: Dict[str, pd.DataFrame], output_dir: str = "./plots"):
    """
    Plot optimizer state memory savings compared to Adam.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if "Adam" not in data:
        print("Adam baseline not found, skipping optimizer savings plot")
        return
    
    adam_opt_memory = extract_component_value(data["Adam"], "optimizer_states_total")
    
    # Extract rSVD data
    rsvd_keys = sorted([k for k in data.keys() if k.startswith('r')], 
                      key=lambda x: int(x[1:]))
    
    if not rsvd_keys:
        print("No rSVD data found for optimizer savings plot")
        return
    
    ranks = []
    opt_memory = []
    savings = []
    
    for key in rsvd_keys:
        ranks.append(int(key[1:]))
        opt_mem = extract_component_value(data[key], "optimizer_states_total")
        opt_memory.append(opt_mem)
        savings.append(adam_opt_memory - opt_mem)
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar chart for optimizer memory
    bars = ax1.bar([f"R{r}" for r in ranks], opt_memory, color='#ff7f0e', alpha=0.7, label='rSVD Optimizer Memory')
    ax1.axhline(y=adam_opt_memory, color='#1f77b4', linestyle='--', linewidth=2, label='Adam Optimizer Memory')
    
    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Optimizer State Memory (MB)', fontsize=12, color='#ff7f0e')
    ax1.tick_params(axis='y', labelcolor='#ff7f0e')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add savings as line plot on second axis
    ax2 = ax1.twinx()
    ax2.plot([f"R{r}" for r in ranks], savings, color='#2ca02c', marker='o', 
             linewidth=2, markersize=8, label='Memory Savings')
    ax2.set_ylabel('Memory Savings vs Adam (MB)', fontsize=12, color='#2ca02c')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.set_title('Optimizer State Memory: rSVD vs Adam', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimizer_memory_savings_breakdown.png"), dpi=300)
    plt.close()
    print("✓ Saved optimizer_memory_savings_breakdown.png")


def plot_cuda_memory_breakdown(data: Dict[str, pd.DataFrame], output_dir: str = "./plots"):
    """
    Plot CUDA memory breakdown (allocated, reserved, peak) across ranks.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    ranks = []
    allocated = []
    reserved = []
    peak_allocated = []
    
    # Sort keys: Adam first, then by rank
    sorted_keys = sorted([k for k in data.keys() if k != "Adam"], 
                        key=lambda x: int(x[1:]) if x.startswith('r') else 999)
    sorted_keys = ["Adam"] + sorted_keys
    
    for key in sorted_keys:
        df = data[key]
        
        if key == "Adam":
            ranks.append("Adam")
        else:
            ranks.append(key.upper())
        
        allocated.append(extract_component_value(df, "cuda_allocated_mb"))
        reserved.append(extract_component_value(df, "cuda_reserved_mb"))
        peak_allocated.append(extract_component_value(df, "cuda_peak_allocated_mb"))
    
    # Create grouped bar chart
    x = range(len(ranks))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar([i - width for i in x], allocated, width, label='Allocated', color='#1f77b4')
    ax.bar(x, reserved, width, label='Reserved', color='#ff7f0e')
    ax.bar([i + width for i in x], peak_allocated, width, label='Peak Allocated', color='#2ca02c')
    
    ax.set_xlabel('Optimizer / Rank', fontsize=12)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('CUDA Memory Breakdown Across Ranks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ranks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cuda_memory_breakdown.png"), dpi=300)
    plt.close()
    print("✓ Saved cuda_memory_breakdown.png")


def main():
    """Main function to generate all memory breakdown plots."""
    print("=" * 80)
    print("Memory Breakdown Plotting")
    print("=" * 80)
    
    # Use current directory if we're already in graph/, otherwise use ./graph
    import os
    if os.path.basename(os.getcwd()) == "graph":
        base_path = "."
    else:
        base_path = "./graph"
    output_dir = "./plots"
    
    # Load all memory breakdown files
    print("\nLoading memory breakdown data...")
    data = load_memory_breakdowns(base_path)
    
    if not data:
        print("No memory breakdown files found!")
        return
    
    print(f"\nFound {len(data)} memory breakdown files")
    
    # Generate all plots
    print("\nGenerating plots...")
    print("-" * 80)
    
    plot_main_components_comparison(data, output_dir)
    plot_stacked_memory_composition(data, output_dir)
    plot_optimizer_state_breakdown(data, output_dir)
    plot_total_memory_comparison(data, output_dir)
    plot_optimizer_savings(data, output_dir)
    plot_cuda_memory_breakdown(data, output_dir)
    
    print("\n" + "=" * 80)
    print("All plots generated successfully!")
    print("=" * 80)
    print(f"\nPlots saved to: {output_dir}/")
    print("  - memory_components_comparison.png")
    print("  - memory_composition_stacked.png")
    print("  - optimizer_state_breakdown.png")
    print("  - total_memory_comparison.png")
    print("  - optimizer_memory_savings_breakdown.png")
    print("  - cuda_memory_breakdown.png")


if __name__ == "__main__":
    main()

