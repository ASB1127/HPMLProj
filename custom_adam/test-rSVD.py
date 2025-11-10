import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from optimizer import CustomAdam  # your optimizer

# ----- Configuration -----
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
epochs = 30
batch_size = 128
lr = 1e-3
compression_interval = 10  # every 10 epochs
rank_fraction = 0.5        # compress to 50% rank

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

model = MLP().to(device)
optimizer = CustomAdam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# ----- Helper for rSVD Compression -----
def apply_rsvd(layer: nn.Linear, rank_fraction: float = 0.5):
    """Apply randomized SVD (torch.svd_lowrank) to a Linear layer's weight."""
    with torch.no_grad():
        W = layer.weight.data
        # Compute target rank
        r = max(1, int(W.shape[1] * rank_fraction))

        # Use torch.svd_lowrank for efficient low-rank approximation
        U, S, Vh = torch.svd_lowrank(W, q=r)

        # Reconstruct compressed weight
        W_compressed = U @ torch.diag(S) @ Vh.T

        # Update layer weight
        layer.weight = nn.Parameter(W_compressed)
        layer.in_features = W_compressed.shape[1]

        print(f"   → compressed {tuple(W.shape)} → {tuple(W_compressed.shape)}")

        return r

# ----- Training & Compression Loop -----
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"[Epoch {epoch:02d}] Train Loss = {avg_loss:.4f}")

    # Apply rSVD compression every N epochs
    if epoch % compression_interval == 0:
        print(f"Applying rSVD compression at epoch {epoch}...")
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                new_in = apply_rsvd(layer, rank_fraction)
        print("Compression complete.\n")

# ----- Evaluation -----
model.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()

accuracy = correct / len(test_loader.dataset)
print(f"\n✅ Final Accuracy: {accuracy * 100:.2f}%")

