import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_SIZE = 10000       # Dataset size
VAL_SIZE = 1000
NUM_EPOCHS = 50000       # Increased epochs to give Grokking more time
LLC_INTERVAL = 100
SGLD_STEPS = 100
BURN_IN = 20
WEIGHT_DECAY = 1e-2    # Encourages simpler (singular) solutions

# ==========================================
# 2. MODEL & SGLD OPTIMIZER
# ==========================================
class GrokkingNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 10)
        )
    def forward(self, x): return self.main(x)

class SGLD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=1.0):
        super().__init__(params, dict(lr=lr, beta=beta))
    def step(self):
        for group in self.param_groups:
            noise_std = np.sqrt(2 * group['lr'] / group['beta'])
            for p in group['params']:
                if p.grad is None: continue
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(p.grad.data, alpha=-group['lr'])
                p.data.add_(noise)

# ==========================================
# 3. EVALUATION FUNCTIONS
# ==========================================
def get_metrics(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item()
            _, pred = torch.max(out.data, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return total_loss / len(loader), 100 * correct / total

def get_llc(model, loader, device, criterion, steps=SGLD_STEPS, burn_in=BURN_IN):
    n = len(loader.dataset)
    beta = 1 / np.log(n)
    model.eval()
    with torch.no_grad():
        w0_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in loader) / len(loader)
    
    optimizer = SGLD(model.parameters(), lr=1e-5, beta=beta)
    sampled_losses = []
    model.train()
    for i in range(steps):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); criterion(model(x), y).backward(); optimizer.step()
            if i > burn_in: sampled_losses.append(criterion(model(x), y).item())
    return (n * beta / 2) * (np.mean(sampled_losses) - w0_loss)

# ==========================================
# 4. DATA & TRAINING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
full_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_idx = torch.randperm(len(full_train))[:DATA_SIZE]
test_idx = torch.randperm(len(full_test))[:VAL_SIZE]
train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_train, train_idx), batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_test, test_idx), batch_size=64)

model = GrokkingNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

history = {'epoch': [], 't_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': [], 'llc': []}

print(f"Tracking Train Loss, Val Loss, and LLC...")
for epoch in range(NUM_EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); criterion(model(x), y).backward(); optimizer.step()
    
    if epoch % LLC_INTERVAL == 0:
        tl, ta = get_metrics(model, train_loader, device, criterion)
        vl, va = get_metrics(model, val_loader, device, criterion)
        llc = get_llc(model, train_loader, device, criterion)
        
        for k, v in zip(history.keys(), [epoch, tl, vl, ta, va, llc]): history[k].append(v)
        print(f"Ep {epoch:03d} | T-Loss: {tl:.3f} | V-Loss: {vl:.3f} | T-Acc: {ta:.1f}% | V-Acc: {va:.1f}% | LLC: {llc:.2f}")

# ==========================================
# 5. VISUALIZATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot Loss (Log Scale to see tiny differences)
ax1.plot(history['epoch'], history['t_loss'], 'g--', label='Train Loss')
ax1.plot(history['epoch'], history['v_loss'], 'b-', label='Val Loss', linewidth=2)
ax1.set_yscale('log')
ax1.set_ylabel('Loss (Log Scale)')
ax1.legend(); ax1.set_title("Training vs Validation Loss")

# Plot LLC and Accuracies
ax2.plot(history['epoch'], history['v_acc'], 'b-', label='Val Acc', linewidth=2)
ax2.plot(history['epoch'], history['t_acc'], 'g--', label='Train Acc')
ax2.set_ylabel('Accuracy (%)')
ax2_twin = ax2.twinx()
ax2_twin.plot(history['epoch'], history['llc'], 'r-o', alpha=0.5, label='LLC')
ax2_twin.set_ylabel('LLC (Complexity)')
ax2.legend(loc='upper left'); ax2_twin.legend(loc='upper right')

plt.tight_layout()
plt.show()