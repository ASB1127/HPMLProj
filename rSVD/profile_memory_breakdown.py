"""
Profile memory usage breakdown for different components of an LLM during training.

This script measures:
1. Model Parameters memory
2. Optimizer States memory
3. Gradients memory
4. Activations memory (during forward pass)
5. Total CUDA memory breakdown

Usage:
    python profile_memory_breakdown.py
"""
import torch
import torch.nn as nn
import os
import csv
from typing import Dict, List, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from optimizer.rSVD_adam_optimizer import rSVDAdam
from utils import get_device, count_parameters


def calculate_parameter_memory(model: nn.Module) -> Dict[str, float]:
    """
    Calculate memory used by model parameters.
    
    Returns:
        Dictionary with total and trainable parameter memory in MB
    """
    total_bytes = 0
    trainable_bytes = 0
    
    for param in model.parameters():
        param_bytes = param.numel() * param.element_size()
        total_bytes += param_bytes
        if param.requires_grad:
            trainable_bytes += param_bytes
    
    return {
        'total_mb': total_bytes / (1024 ** 2),
        'trainable_mb': trainable_bytes / (1024 ** 2),
        'total_bytes': total_bytes,
        'trainable_bytes': trainable_bytes
    }


def calculate_optimizer_state_memory(optimizer) -> Dict[str, float]:
    """
    Calculate memory used by optimizer states.
    
    Returns:
        Dictionary with optimizer state memory in MB
    """
    total_bytes = 0
    state_breakdown = {}
    
    for param in optimizer.param_groups[0]['params']:
        if param in optimizer.state:
            state = optimizer.state[param]
            param_bytes = 0
            
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    tensor_bytes = value.numel() * value.element_size()
                    param_bytes += tensor_bytes
                    
                    # Track by state key (e.g., exp_avg, exp_avg_sq, proj, etc.)
                    if key not in state_breakdown:
                        state_breakdown[key] = 0
                    state_breakdown[key] += tensor_bytes
            
            total_bytes += param_bytes
    
    return {
        'total_mb': total_bytes / (1024 ** 2),
        'total_bytes': total_bytes,
        'breakdown': {k: v / (1024 ** 2) for k, v in state_breakdown.items()}
    }


def calculate_gradient_memory(model: nn.Module) -> Dict[str, float]:
    """
    Calculate memory used by gradients.
    
    Returns:
        Dictionary with gradient memory in MB
    """
    total_bytes = 0
    
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            total_bytes += param.grad.numel() * param.grad.element_size()
    
    return {
        'total_mb': total_bytes / (1024 ** 2),
        'total_bytes': total_bytes
    }


class ActivationMemoryTracker:
    """
    Track activation memory during forward pass using hooks.
    """
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
        self.activation_bytes = 0
        self.hooks = []
        self.layer_names = []
        
    def _hook_fn(self, name):
        """Hook function to track activation memory."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                bytes_used = output.numel() * output.element_size()
                self.activation_bytes += bytes_used
                self.layer_names.append(name)
        return hook
    
    def register_hooks(self):
        """Register forward hooks on all modules."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def reset(self):
        """Reset activation memory counter."""
        self.activation_bytes = 0
        self.layer_names = []
    
    def get_memory(self) -> Dict[str, float]:
        """Get tracked activation memory."""
        return {
            'total_mb': self.activation_bytes / (1024 ** 2),
            'total_bytes': self.activation_bytes
        }


def get_cuda_memory_breakdown(device: str) -> Dict[str, float]:
    """
    Get detailed CUDA memory breakdown using torch.cuda.memory_stats().
    
    Returns:
        Dictionary with various memory metrics in MB
    """
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return {}
    
    stats = torch.cuda.memory_stats(device)
    
    return {
        'allocated_mb': stats['allocated_bytes.all.current'] / (1024 ** 2),
        'reserved_mb': stats['reserved_bytes.all.current'] / (1024 ** 2),
        'active_mb': stats['active_bytes.all.current'] / (1024 ** 2),
        'inactive_split_mb': stats['inactive_split_bytes.all.current'] / (1024 ** 2),
        'peak_allocated_mb': stats['allocated_bytes.all.peak'] / (1024 ** 2),
        'peak_reserved_mb': stats['reserved_bytes.all.peak'] / (1024 ** 2),
    }


def profile_training_step_memory(
    model: nn.Module,
    optimizer,
    batch: Dict[str, torch.Tensor],
    device: str,
    rank: Optional[int] = None,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 4
) -> Dict[str, any]:
    """
    Profile memory usage during training with gradient accumulation.
    
    Args:
        model: The model to profile
        optimizer: The optimizer
        batch: Input batch dictionary (should be batch_size=32 to match training)
        device: Device string
        rank: Optional rank identifier for rSVD
        batch_size: Batch size used (for reporting)
        gradient_accumulation_steps: Number of gradient accumulation steps (default 4, matching training)
        
    Returns:
        Dictionary with memory breakdown for all components
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    results = {
        'rank': rank,
        'device': device,
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps
    }
    
    # 1. Model Parameters Memory
    param_mem = calculate_parameter_memory(model)
    results['parameters'] = param_mem
    print(f"\n[Parameters] Total: {param_mem['total_mb']:.2f} MB, "
          f"Trainable: {param_mem['trainable_mb']:.2f} MB")
    
    # 2. Optimizer States Memory
    opt_mem = calculate_optimizer_state_memory(optimizer)
    results['optimizer_states'] = opt_mem
    print(f"[Optimizer States] Total: {opt_mem['total_mb']:.2f} MB")
    if opt_mem['breakdown']:
        print("  Breakdown:")
        for key, mb in sorted(opt_mem['breakdown'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {key}: {mb:.2f} MB")
    
    # 3. Forward Pass - Track Activations (single batch)
    # Use model.train() to match actual training context
    model.train()  # Match training mode (affects dropout, batch norm, etc.)
    activation_tracker = ActivationMemoryTracker(model, device)
    activation_tracker.register_hooks()
    
    optimizer.zero_grad()
    
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Move batch to device
    inputs = {k: v.to(device) for k, v in batch.items() 
              if k in ['input_ids', 'attention_mask']}
    labels = batch['labels'].to(device) if 'labels' in batch else None
    
    # Forward pass (with gradients enabled to match training)
    # Track activations during this forward pass (single batch)
    outputs = model(**inputs)
    
    # Get activation memory before backward pass
    activation_mem = activation_tracker.get_memory()
    results['activations'] = activation_mem
    print(f"[Activations] Total: {activation_mem['total_mb']:.2f} MB (batch_size={batch_size}, single batch)")
    
    # Remove hooks before backward (they're no longer needed)
    activation_tracker.remove_hooks()
    
    # 4. Gradient Accumulation - Simulate multiple steps
    # This matches training behavior where gradients accumulate across multiple batches
    print(f"\n[Gradient Accumulation] Running {gradient_accumulation_steps} accumulation steps...")
    
    # First backward pass (activations from above forward pass)
    if labels is not None:
        loss = nn.functional.cross_entropy(outputs.logits, labels)
    else:
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
    
    # Scale loss by accumulation steps (standard practice)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    
    # Additional accumulation steps (reuse same batch for simplicity)
    # In real training, these would be different batches, but memory pattern is similar
    for step in range(1, gradient_accumulation_steps):
        # Forward pass for this accumulation step
        outputs = model(**inputs)
        if labels is not None:
            loss = nn.functional.cross_entropy(outputs.logits, labels)
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Scale and accumulate gradients
        loss = loss / gradient_accumulation_steps
        loss.backward()
    
    # Now measure gradients after all accumulation steps
    # All gradients from gradient_accumulation_steps batches are in memory
    grad_mem = calculate_gradient_memory(model)
    results['gradients'] = grad_mem
    print(f"[Gradients] Total: {grad_mem['total_mb']:.2f} MB (after {gradient_accumulation_steps} accumulation steps)")
    
    # Note: In real training, optimizer.step() would be called here
    # For profiling, we measure before the step to see accumulated gradient memory
    
    # 5. CUDA Memory Breakdown
    cuda_breakdown = get_cuda_memory_breakdown(device)
    results['cuda_breakdown'] = cuda_breakdown
    if cuda_breakdown:
        print(f"\n[CUDA Memory Breakdown]")
        print(f"  Allocated: {cuda_breakdown['allocated_mb']:.2f} MB")
        print(f"  Reserved: {cuda_breakdown['reserved_mb']:.2f} MB")
        print(f"  Active: {cuda_breakdown['active_mb']:.2f} MB")
        print(f"  Peak Allocated: {cuda_breakdown['peak_allocated_mb']:.2f} MB")
        print(f"  Peak Reserved: {cuda_breakdown['peak_reserved_mb']:.2f} MB")
    
    # 6. Total Summary
    # Note: Activations are measured for a single batch, but gradients include accumulation
    # In real training, activations from all accumulation batches would be in memory
    # but they're processed sequentially, so peak is typically from one batch's activations
    total_components = (
        param_mem['total_mb'] +
        opt_mem['total_mb'] +
        activation_mem['total_mb'] +
        grad_mem['total_mb']
    )
    results['total_components_mb'] = total_components
    
    print(f"\n[Total Components] {total_components:.2f} MB")
    print(f"  (Parameters + Optimizer + Activations + Gradients)")
    print(f"  Note: Activations measured for 1 batch, gradients accumulated over {gradient_accumulation_steps} steps")
    
    if cuda_breakdown:
        print(f"[CUDA Allocated] {cuda_breakdown['allocated_mb']:.2f} MB")
        print(f"  (Difference: {cuda_breakdown['allocated_mb'] - total_components:.2f} MB)")
        print(f"  (This includes overhead, temporary buffers, etc.)")
    
    return results


def save_memory_breakdown(results: Dict, output_dir: str, rank: Optional[int] = None):
    """
    Save memory breakdown results to CSV files.
    
    Args:
        results: Memory breakdown results dictionary
        output_dir: Directory to save CSV files
        rank: Optional rank identifier
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine filename based on rank
    if rank is not None:
        filename = f"memory_breakdown_r{rank}.csv"
    else:
        filename = "memory_breakdown.csv"
    
    csv_path = os.path.join(output_dir, filename)
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "memory_mb", "memory_bytes", "details"])
        # Add batch size and gradient accumulation info
        if 'batch_size' in results:
            writer.writerow([
                "batch_size",
                results['batch_size'],
                results['batch_size'],
                f"Batch size used for profiling"
            ])
        if 'gradient_accumulation_steps' in results:
            writer.writerow([
                "gradient_accumulation_steps",
                results['gradient_accumulation_steps'],
                results['gradient_accumulation_steps'],
                f"Number of gradient accumulation steps (effective batch = {results['batch_size'] * results['gradient_accumulation_steps']})"
            ])
        
        # Parameters
        writer.writerow([
            "parameters_total",
            results['parameters']['total_mb'],
            results['parameters']['total_bytes'],
            ""
        ])
        writer.writerow([
            "parameters_trainable",
            results['parameters']['trainable_mb'],
            results['parameters']['trainable_bytes'],
            ""
        ])
        
        # Optimizer States
        writer.writerow([
            "optimizer_states_total",
            results['optimizer_states']['total_mb'],
            results['optimizer_states']['total_bytes'],
            ""
        ])
        for key, mb in results['optimizer_states']['breakdown'].items():
            writer.writerow([
                f"optimizer_state_{key}",
                mb,
                int(mb * (1024 ** 2)),
                ""
            ])
        
        # Activations
        writer.writerow([
            "activations_total",
            results['activations']['total_mb'],
            results['activations']['total_bytes'],
            ""
        ])
        
        # Gradients
        writer.writerow([
            "gradients_total",
            results['gradients']['total_mb'],
            results['gradients']['total_bytes'],
            ""
        ])
        
        # CUDA Breakdown
        if results.get('cuda_breakdown'):
            for key, mb in results['cuda_breakdown'].items():
                writer.writerow([
                    f"cuda_{key}",
                    mb,
                    int(mb * (1024 ** 2)),
                    ""
                ])
        
        # Total
        writer.writerow([
            "total_components",
            results['total_components_mb'],
            int(results['total_components_mb'] * (1024 ** 2)),
            "Sum of Parameters + Optimizer + Activations + Gradients"
        ])
    
    print(f"\n✓ Saved memory breakdown to {csv_path}")


def main():
    """Main function to profile memory breakdown."""
    device = get_device()
    print("=" * 80)
    print("LLM Memory Breakdown Profiler")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Batch Size: 32 (matching training configuration)")
    print(f"Gradient Accumulation Steps: 4 (matching training configuration)")
    print(f"Effective Batch Size: 128 (32 × 4)")
    print(f"Note: This profiler now matches training config exactly\n")
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    num_labels = 2
    
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get parameter info
    param_info = count_parameters(model)
    print(f"Total parameters: {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")
    
    # Load dataset with enough samples for batch_size=32 and gradient accumulation
    print("\nLoading dataset...")
    batch_size = 32
    gradient_accumulation_steps = 4  # Match training configuration
    dataset = load_dataset("sst2", split="train[:128]")  # Load enough for multiple batches
    
    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    tokenized_ds = tokenized_ds.rename_column("label", "labels")
    tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Create a DataLoader to get proper batches (batch_size=32, matching training)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(tokenized_ds, batch_size=batch_size, shuffle=False)
    
    # Get a single batch of size 32 (will be reused for gradient accumulation)
    batch = next(iter(dataloader))
    print(f"Using batch size: {batch['input_ids'].shape[0]} (matching training config)")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps} (matching training config)")
    
    # Test with regular Adam
    print("\n" + "=" * 80)
    print("1. Profiling with Regular Adam Optimizer")
    print("=" * 80)
    
    model_adam = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    ).to(device)
    
    adam_optimizer = torch.optim.Adam(model_adam.parameters(), lr=1e-4)
    
    # Initialize optimizer state
    for param in model_adam.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param)
    adam_optimizer.step()
    
    results_adam = profile_training_step_memory(
        model_adam, adam_optimizer, batch, device, rank=None, 
        batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    output_dir = "./graph"
    save_memory_breakdown(results_adam, output_dir, rank=None)
    
    # Cleanup
    del model_adam, adam_optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    # Test with rSVD Adam for different ranks
    ranks = [4, 8, 16, 64, 128]
    
    for rank in ranks:
        print("\n" + "=" * 80)
        print(f"2. Profiling with rSVD Adam (rank={rank})")
        print("=" * 80)
        
        model_rsvd = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(device)
        
        rank_fraction = rank / 768.0  # Typical dimension for DistilBERT
        rsvd_optimizer = rSVDAdam(
            model_rsvd.parameters(),
            lr=1e-4,
            rank_fraction=rank_fraction,
            proj_interval=500,
            use_rgp=True,
            verbose_memory_once=False
        )
        
        # Initialize optimizer state
        for param in model_rsvd.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param)
        rsvd_optimizer.step()
        
        results_rsvd = profile_training_step_memory(
            model_rsvd, rsvd_optimizer, batch, device, rank=rank, 
            batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        save_memory_breakdown(results_rsvd, output_dir, rank=rank)
        
        # Cleanup
        del model_rsvd, rsvd_optimizer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("Memory Profiling Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print("  - memory_breakdown.csv (Adam baseline)")
    print("  - memory_breakdown_r{rank}.csv (rSVD for each rank)")


if __name__ == "__main__":
    main()

