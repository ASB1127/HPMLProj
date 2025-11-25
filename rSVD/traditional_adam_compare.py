import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity, record_function

# --------------------------------------------------------
# Import rSVDAdam (with rSVD compression)
# --------------------------------------------------------
from optimizer import rSVDAdam
from utils import (
    get_device, reset_memory_stats, get_peak_memory_mb,
    extract_timing_stats, get_profiler_activities,
    format_memory_comparison, format_timing_comparison
)

# ----- Configuration -----
device = get_device()
epochs = 15
batch_size = 128
lr = 1e-3

# ----- Dataset -----
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(".", train=True, download=True, transform=transform)
test_data = datasets.MNIST(".", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=512)

# ----- Model -----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ----- Helper Function with Profiling -----
def train(model, optimizer, label, profile_batches=5):
    """
    Train model with profiling enabled for first profile_batches batches only.
    
    Args:
        profile_batches: Number of batches to profile (profiles only first epoch)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    
    # Reset memory stats
    reset_memory_stats(device)
    
    print(f"[{label}] Starting training...")
    
    # Warmup: run one batch without profiling
    model.train()
    warmup_batch = next(iter(train_loader))
    x_warmup, y_warmup = warmup_batch[0].to(device), warmup_batch[1].to(device)
    optimizer.zero_grad()
    out = model(x_warmup)
    loss = criterion(out, y_warmup)
    loss.backward()
    optimizer.step()
    
    # Profile only the first few batches of first epoch
    profile_results = None
    activities = get_profiler_activities(device)
    
    # Profile only first profile_batches batches to save memory
    prof = None
    with profile(
        activities=activities,
        record_shapes=False,  # Disable to save memory
        profile_memory=True,
        with_stack=False  # Disable to save memory
    ) as prof:
        model.train()
        batch_count = 0
        
        # Profile only first profile_batches batches
        for x, y in train_loader:
            if batch_count >= profile_batches:
                break  # Exit profiler context early
                
            x, y = x.to(device), y.to(device)
            with record_function(f"{label}_batch_{batch_count}"):
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            batch_count += 1
    
    # Extract profiling results immediately
    peak_mem = get_peak_memory_mb(device)
    if peak_mem is not None:
        profile_results = {
            'peak_memory_mb': peak_mem,
            'profiler': prof
        }
    else:
        profile_results = {'profiler': prof}
    
    # Clear profiler from memory immediately
    prof = None
    reset_memory_stats(device)
    
    # Continue training all epochs normally (without profiling)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        print(f"[{label}] Epoch {epoch:02d} | Train Loss = {avg_loss:.4f}")

    # Evaluate accuracy
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f"[{label}] Final Accuracy = {acc * 100:.2f}%\n")

    return train_losses, acc, profile_results

# ============================================================
# 1️⃣ Train with PyTorch Adam
# ============================================================
print("=" * 70)
print("Training with PyTorch Adam")
print("=" * 70)
model1 = MLP()
adam_opt = torch.optim.Adam(model1.parameters(), lr=lr)
adam_losses, adam_acc, adam_profile = train(model1, adam_opt, "Adam", profile_batches=5)

# Clear memory
del model1, adam_opt
reset_memory_stats(device)

# ============================================================
# 2️⃣ Train with rSVDAdam (with rSVD)
# ============================================================
print("=" * 70)
print("Training with rSVDAdam")
print("=" * 70)
model2 = MLP()
rgp_opt = rSVDAdam(
    model2.parameters(),
    lr=lr,
    rank_fraction=0.25,
    proj_interval=200,
    use_rgp=True,
    verbose_memory_once=True
)
rgp_losses, rgp_acc, rgp_profile = train(model2, rgp_opt, "rSVDAdam", profile_batches=5)

# ============================================================
# 📊 Performance Analysis
# ============================================================
print("\n" + "=" * 70)
print("Performance Comparison: Adam vs rSVDAdam")
print("=" * 70)

# Memory comparison
if adam_profile and rgp_profile:
    adam_mem = adam_profile.get('peak_memory_mb', 0)
    rgp_mem = rgp_profile.get('peak_memory_mb', 0)
    if adam_mem > 0 and rgp_mem > 0:
        print(f"\n📊 Memory Usage (Peak):")
        print(format_memory_comparison("Adam", adam_mem, "rSVDAdam", rgp_mem))

# Timing comparison from profiler
print(f"\n⏱️  Timing Analysis (from profiled batches):")
print("-" * 70)

# Get timing stats from profiler
adam_timing = extract_timing_stats(
    adam_profile['profiler'] if adam_profile else None,
    device, "Adam"
)
rgp_timing = extract_timing_stats(
    rgp_profile['profiler'] if rgp_profile else None,
    device, "rSVDAdam"
)

if adam_timing and rgp_timing:
    num_profiled_batches = 5  # Match profile_batches parameter
    if device.startswith("cuda") and torch.cuda.is_available():
        if 'cuda_time_ms' in adam_timing and 'cuda_time_ms' in rgp_timing:
            adam_cuda = adam_timing['cuda_time_ms'] / num_profiled_batches
            rgp_cuda = rgp_timing['cuda_time_ms'] / num_profiled_batches
            print(f"  CUDA Time (per batch, averaged over {num_profiled_batches} batches):")
            print(format_timing_comparison("Adam", adam_cuda, "rSVDAdam", rgp_cuda))
    
    if 'cpu_time_ms' in adam_timing and 'cpu_time_ms' in rgp_timing:
        adam_cpu = adam_timing['cpu_time_ms'] / num_profiled_batches
        rgp_cpu = rgp_timing['cpu_time_ms'] / num_profiled_batches
        print(f"  CPU Time (per batch, averaged over {num_profiled_batches} batches):")
        print(format_timing_comparison("Adam", adam_cpu, "rSVDAdam", rgp_cpu))

# Print detailed profiler table for optimizer step
print(f"\n📋 Detailed Profiler Output (Top 10 operations by CUDA time):")
print("-" * 70)
if adam_profile and device.startswith("cuda") and torch.cuda.is_available():
    print("\nAdam Optimizer:")
    print(adam_profile['profiler'].key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=10,
        max_name_column_width=50
    ))

if rgp_profile and device.startswith("cuda") and torch.cuda.is_available():
    print("\nrSVDAdam Optimizer:")
    print(rgp_profile['profiler'].key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=10,
        max_name_column_width=50
    ))

# Memory breakdown
if device.startswith("cuda") and torch.cuda.is_available():
    print(f"\n💾 Memory Breakdown (Top 10 operations by CUDA memory):")
    print("-" * 70)
    if adam_profile:
        print("\nAdam Optimizer:")
        print(adam_profile['profiler'].key_averages().table(
            sort_by="cuda_memory_usage", 
            row_limit=10,
            max_name_column_width=50
        ))
    
    if rgp_profile:
        print("\nrSVDAdam Optimizer:")
        print(rgp_profile['profiler'].key_averages().table(
            sort_by="cuda_memory_usage", 
            row_limit=10,
            max_name_column_width=50
        ))

# ============================================================
# 📊 Plot comparison
# ============================================================
plt.figure(figsize=(7, 5))
plt.plot(adam_losses, label=f"Adam ({adam_acc*100:.2f}% acc)", marker="o")
plt.plot(rgp_losses, label=f"rSVDAdam ({rgp_acc*100:.2f}% acc)", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss: Adam vs rSVDAdam (with rSVD)")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("loss_comparison.png", dpi=300, bbox_inches="tight")
