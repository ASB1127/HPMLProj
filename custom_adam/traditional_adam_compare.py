import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Import your CustomAdam (no rSVD or compression)
# --------------------------------------------------------
from optimizer import CustomAdam  

# ----- Configuration -----
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
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

# ----- Helper Function -----
def train(model, optimizer, label):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_losses = []

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

    return train_losses, acc

# ============================================================
# 1️⃣ Train with PyTorch Adam
# ============================================================
model1 = MLP()
adam_opt = torch.optim.Adam(model1.parameters(), lr=lr)
adam_losses, adam_acc = train(model1, adam_opt, "Adam")

# ============================================================
# 2️⃣ Train with CustomAdam (no rSVD)
# ============================================================
model2 = MLP()
custom_opt = CustomAdam(model2.parameters(), lr=lr)
custom_losses, custom_acc = train(model2, custom_opt, "CustomAdam")

# ============================================================
# 📊 Plot comparison
# ============================================================
plt.figure(figsize=(7, 5))
plt.plot(adam_losses, label=f"Adam ({adam_acc*100:.2f}% acc)", marker="o")
plt.plot(custom_losses, label=f"CustomAdam ({custom_acc*100:.2f}% acc)", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss: Adam vs CustomAdam (no rSVD)")
plt.legend()
plt.grid(True)
plt.show()

