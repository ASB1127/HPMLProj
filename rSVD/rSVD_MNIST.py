import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from optimizer import rSVDAdam
from utils import get_device

# ----- Configuration -----
device = get_device()
epochs = 10
batch_size = 128
lr = 1e-3
rank_fraction = 0.25       # fraction of min(m, n) to use as target rank
proj_interval = 200        # recompute projector P every N steps

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
optimizer = rSVDAdam(
    model.parameters(),
    lr=lr,
    rank_fraction=rank_fraction,
    proj_interval=proj_interval,
    use_rgp=True,
    verbose_memory_once=True
)
criterion = nn.CrossEntropyLoss()

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

