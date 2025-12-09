"""
Compare optimizer state memory between regular Adam and rSVD Adam.

This script measures ONLY the optimizer state memory (exp_avg, exp_avg_sq, etc.),
not the full training memory.
"""
import torch
import torch.nn as nn
import os
import csv
from transformers import AutoModelForSequenceClassification
from optimizer.rSVD_adam_optimizer import rSVDAdam
from utils import get_device


def _get_rank_dir(rank, base_path="./graph"):
    """Helper function to create consistent directory name from rank."""
    return f"{base_path}/r{rank}_opt"


def _rank_to_rank_fraction(rank, typical_dim=768):
    """
    Convert rank (integer) to rank_fraction for optimizer.
    
    For DistilBERT, typical attention dimensions are ~768, so we estimate
    rank_fraction = rank / typical_dim. The optimizer will then calculate
    the actual rank as min(rank_fraction * min(m,n), rank) per parameter.
    
    Args:
        rank: Target rank (integer)
        typical_dim: Typical dimension for the model (default 768 for DistilBERT)
    
    Returns:
        rank_fraction: Fraction to pass to optimizer
    """
    return rank / typical_dim


def calculate_optimizer_state_memory(optimizer):
    """
    Calculate the total memory used by optimizer state.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Total memory in bytes and MB
    """
    total_bytes = 0
    state_info = []
    
    for param_id, param in enumerate(optimizer.param_groups[0]['params']):
        if param in optimizer.state:
            state = optimizer.state[param]
            param_bytes = 0
            
            # Sum memory for all state tensors
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    tensor_bytes = value.numel() * value.element_size()
                    param_bytes += tensor_bytes
                    state_info.append({
                        'param_id': param_id,
                        'param_shape': tuple(param.shape),
                        'state_key': key,
                        'state_shape': tuple(value.shape),
                        'bytes': tensor_bytes
                    })
            
            total_bytes += param_bytes
    
    total_mb = total_bytes / (1024 ** 2)
    return total_bytes, total_mb, state_info


def compare_optimizer_memory(model_name="distilbert-base-uncased", num_labels=2, 
                            ranks=[4, 8, 16, 64, 128], device="cuda", typical_dim=768):
    """
    Compare optimizer memory between regular Adam and rSVD Adam.
    
    Args:
        model_name: Name of the model to use
        num_labels: Number of labels for classification
        ranks: List of ranks (integers) to test for rSVD
        device: Device to use
        typical_dim: Typical dimension for rank_fraction conversion (default 768 for DistilBERT)
    """
    print("=" * 80)
    print("Optimizer State Memory Comparison")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Device: {device}\n")
    
    results = {}
    
    # =========================================
    # 1. Regular Adam
    # =========================================
    print("-" * 80)
    print("1. Regular Adam Optimizer")
    print("-" * 80)
    
    # Create a fresh model instance for Adam
    model_adam = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model_adam.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model_adam.parameters() if p.requires_grad):,}")
    
    # Create dummy gradients to initialize optimizer state
    for param in model_adam.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param)
    
    adam_optimizer = torch.optim.Adam(model_adam.parameters(), lr=1e-4)
    adam_optimizer.step()  # Initialize optimizer state
    
    adam_bytes, adam_mb, adam_state_info = calculate_optimizer_state_memory(adam_optimizer)
    
    print(f"Optimizer State Memory: {adam_mb:.2f} MB ({adam_bytes:,} bytes)")
    print(f"\nState breakdown:")
    for info in adam_state_info[:10]:  # Show first 10
        print(f"  Param {info['param_id']} {info['param_shape']}: "
              f"{info['state_key']} {info['state_shape']} = {info['bytes']/(1024**2):.4f} MB")
    if len(adam_state_info) > 10:
        print(f"  ... and {len(adam_state_info) - 10} more state tensors")
    
    results['Adam'] = {
        'memory_mb': adam_mb,
        'memory_bytes': adam_bytes,
        'state_info': adam_state_info
    }
    
    # Cleanup
    del model_adam, adam_optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # =========================================
    # 2. rSVD Adam with different ranks
    # =========================================
    for rank in ranks:
        print("\n" + "-" * 80)
        print(f"2. rSVD Adam (rank={rank})")
        print("-" * 80)
        
        # Convert rank to rank_fraction for optimizer
        rank_fraction = _rank_to_rank_fraction(rank, typical_dim)
        print(f"   Using rank_fraction={rank_fraction:.4f} (rank={rank} / typical_dim={typical_dim})")
        
        # Create a fresh model instance for rSVD
        model_rsvd = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(device)
        
        # Create dummy gradients to initialize optimizer state
        for param in model_rsvd.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param)
        
        rsvd_optimizer = rSVDAdam(
            model_rsvd.parameters(),
            lr=1e-4,
            rank_fraction=rank_fraction,
            proj_interval=500,
            use_rgp=True,
            verbose_memory_once=False  # We'll print our own summary
        )
        rsvd_optimizer.step()  # Initialize optimizer state
        
        rsvd_bytes, rsvd_mb, rsvd_state_info = calculate_optimizer_state_memory(rsvd_optimizer)
        
        print(f"Optimizer State Memory: {rsvd_mb:.2f} MB ({rsvd_bytes:,} bytes)")
        savings = adam_mb - rsvd_mb
        savings_pct = (savings / adam_mb * 100) if adam_mb > 0 else 0
        print(f"Memory Savings vs Adam: {savings:.2f} MB ({savings_pct:.1f}% reduction)")
        
        print(f"\nState breakdown:")
        for info in rsvd_state_info[:10]:  # Show first 10
            print(f"  Param {info['param_id']} {info['param_shape']}: "
                  f"{info['state_key']} {info['state_shape']} = {info['bytes']/(1024**2):.4f} MB")
        if len(rsvd_state_info) > 10:
            print(f"  ... and {len(rsvd_state_info) - 10} more state tensors")
        
        results[f'rSVD_r{rank}'] = {
            'memory_mb': rsvd_mb,
            'memory_bytes': rsvd_bytes,
            'state_info': rsvd_state_info,
            'savings_mb': savings,
            'savings_pct': savings_pct,
            'rank': rank
        }
        
        # Cleanup
        del model_rsvd, rsvd_optimizer
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # =========================================
    # 3. Summary Comparison
    # =========================================
    print("\n" + "=" * 80)
    print("Summary Comparison")
    print("=" * 80)
    print(f"\n{'Optimizer':<20} {'Memory (MB)':<15} {'Savings (MB)':<15} {'Savings (%)':<15}")
    print("-" * 80)
    print(f"{'Adam':<20} {results['Adam']['memory_mb']:<15.2f} {'-':<15} {'-':<15}")
    
    for key in sorted([k for k in results.keys() if k != 'Adam'], 
                      key=lambda x: results[x].get('rank', 0)):
        rsvd_result = results[key]
        print(f"{key:<20} {rsvd_result['memory_mb']:<15.2f} "
              f"{rsvd_result['savings_mb']:<15.2f} {rsvd_result['savings_pct']:<15.1f}")
    
    # =========================================
    # 4. Save Results to CSV Files
    # =========================================
    base_path = "./graph"
    os.makedirs(base_path, exist_ok=True)
    
    # Save Adam baseline
    adam_dir = os.path.join(base_path, "adam_opt")
    os.makedirs(adam_dir, exist_ok=True)
    adam_csv = os.path.join(adam_dir, "optimizer_memory.csv")
    with open(adam_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer", "memory_mb", "memory_bytes", "savings_mb", "savings_pct"])
        writer.writerow(["Adam", results['Adam']['memory_mb'], 
                        results['Adam']['memory_bytes'], 0, 0])
    print(f"\n✓ Saved Adam optimizer memory data to {adam_csv}")
    
    # Save rSVD results for each rank
    for key in sorted([k for k in results.keys() if k != 'Adam'], 
                      key=lambda x: results[x].get('rank', 0)):
        rank = results[key]['rank']
        rank_dir = _get_rank_dir(rank, base_path)
        os.makedirs(rank_dir, exist_ok=True)
        
        rsvd_csv = os.path.join(rank_dir, "optimizer_memory.csv")
        with open(rsvd_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["optimizer", "rank", "memory_mb", "memory_bytes", 
                           "savings_mb", "savings_pct"])
            writer.writerow([
                "rSVDAdam",
                rank,
                results[key]['memory_mb'],
                results[key]['memory_bytes'],
                results[key]['savings_mb'],
                results[key]['savings_pct']
            ])
        print(f"✓ Saved {key} optimizer memory data to {rsvd_csv}")
    
    # Save comparison summary
    summary_csv = os.path.join(base_path, "optimizer_memory_comparison.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer", "rank", "memory_mb", "memory_bytes", 
                        "savings_mb", "savings_pct"])
        writer.writerow(["Adam", "", results['Adam']['memory_mb'], 
                        results['Adam']['memory_bytes'], 0, 0])
        for key in sorted([k for k in results.keys() if k != 'Adam'], 
                          key=lambda x: results[x].get('rank', 0)):
            rank = results[key]['rank']
            rsvd_result = results[key]
            writer.writerow([
                "rSVDAdam",
                rank,
                rsvd_result['memory_mb'],
                rsvd_result['memory_bytes'],
                rsvd_result['savings_mb'],
                rsvd_result['savings_pct']
            ])
    print(f"✓ Saved comparison summary to {summary_csv}")
    
    return results


if __name__ == "__main__":
    device = get_device()
    
    # Use DistilBERT for comparison (same as in rsvd_call.py)
    model_name = "distilbert-base-uncased"
    num_labels = 2
    
    # Test with different ranks (similar to Lora)
    ranks = [4, 8, 16, 64, 128]
    
    results = compare_optimizer_memory(
        model_name=model_name,
        num_labels=num_labels,
        ranks=ranks,
        device=device
    )
    
    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)

