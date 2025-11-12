"""
Common utilities for training scripts.
"""
import torch
from typing import Optional, Dict, Any
from torch.profiler import ProfilerActivity


def get_device() -> str:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def reset_memory_stats(device: str) -> None:
    """Reset peak memory statistics for GPU."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_current_memory_mb(device: str) -> Optional[float]:
    """
    Get current GPU memory usage in MB.
    
    Args:
        device: Device string
        
    Returns:
        Current memory in MB, or None if not CUDA
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return None


def get_peak_memory_mb(device: str) -> Optional[float]:
    """
    Get peak GPU memory usage in MB.
    
    Args:
        device: Device string
        
    Returns:
        Peak memory in MB, or None if not CUDA
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None


def get_reserved_memory_mb(device: str) -> Optional[float]:
    """
    Get reserved GPU memory in MB.
    
    Args:
        device: Device string
        
    Returns:
        Reserved memory in MB, or None if not CUDA
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 ** 2)
    return None


def print_memory_status(device: str, label: str = "") -> None:
    """
    Print current GPU memory status.
    
    Args:
        device: Device string
        label: Optional label for the print statement
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        current = get_current_memory_mb(device)
        peak = get_peak_memory_mb(device)
        reserved = get_reserved_memory_mb(device)
        print(f"[Memory {label}] "
              f"Allocated: {current:.2f} MB | "
              f"Reserved: {reserved:.2f} MB | "
              f"Peak: {peak:.2f} MB")
    else:
        print(f"[Memory {label}] GPU memory tracking not available for {device}")


def extract_timing_stats(prof, device: str, label: str = "") -> Optional[Dict[str, Any]]:
    """
    Extract timing statistics from PyTorch profiler.
    
    Args:
        prof: PyTorch profiler object
        device: Device string
        label: Optional label for error messages
        
    Returns:
        Dictionary with timing stats, or None if extraction fails
    """
    if prof is None:
        return None
    
    try:
        events = prof.key_averages(group_by_input_shape=True)
        
        if device.startswith("cuda") and torch.cuda.is_available():
            cuda_time = sum([e.cuda_time_total for e in events]) / 1e6  # Convert to ms
            cpu_time = sum([e.cpu_time_total for e in events]) / 1e6
            return {
                'cuda_time_ms': cuda_time,
                'cpu_time_ms': cpu_time,
                'events': events
            }
        else:
            cpu_time = sum([e.cpu_time_total for e in events]) / 1e6
            return {
                'cpu_time_ms': cpu_time,
                'events': events
            }
    except Exception as e:
        print(f"  Warning: Could not extract timing for {label}: {e}")
        return None


def get_profiler_activities(device: str):
    """
    Get appropriate profiler activities for the device.
    
    Args:
        device: Device string
        
    Returns:
        List of ProfilerActivity
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        return [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    else:
        return [ProfilerActivity.CPU]


def format_memory_comparison(
    name1: str, mem1: float,
    name2: str, mem2: float,
    unit: str = "MB"
) -> str:
    """
    Format a memory comparison string.
    
    Args:
        name1: First optimizer name
        mem1: First optimizer memory usage
        name2: Second optimizer name
        mem2: Second optimizer memory usage
        unit: Memory unit (default: "MB")
        
    Returns:
        Formatted comparison string
    """
    savings = mem1 - mem2
    savings_pct = (savings / mem1 * 100) if mem1 > 0 else 0
    return (
        f"  {name1}:    {mem1:.2f} {unit}\n"
        f"  {name2}: {mem2:.2f} {unit}\n"
        f"  Savings: {savings:.2f} {unit} ({savings_pct:.1f}% reduction)"
    )


def format_timing_comparison(
    name1: str, time1: float,
    name2: str, time2: float,
    unit: str = "ms",
    label: str = "Time"
) -> str:
    """
    Format a timing comparison string.
    
    Args:
        name1: First optimizer name
        time1: First optimizer time
        name2: Second optimizer name
        time2: Second optimizer time
        unit: Time unit (default: "ms")
        label: Label for the comparison (default: "Time")
        
    Returns:
        Formatted comparison string
    """
    overhead = ((time2 - time1) / time1 * 100) if time1 > 0 else 0
    return (
        f"  {label} (per batch):\n"
        f"    {name1}:    {time1:.2f} {unit}\n"
        f"    {name2}: {time2:.2f} {unit}\n"
        f"    Overhead: {overhead:+.1f}%"
    )


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with 'total' and 'trainable' parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable
    }

