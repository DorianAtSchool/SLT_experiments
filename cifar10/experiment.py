"An attempt to observe phase transitions in training neural networks on CIFAR-10 dataset using LLC metric. "

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
DATA_SIZE = 1000       # Try: 100, 500, 1000, 5000
NUM_EPOCHS = 50
LLC_INTERVAL = 2      # Calculate LLC every X epochs
HIDDEN_DIM = 128      # Model complexity

# ==========================================
# 2. MODEL & SGLD OPTIMIZER
# ==========================================
class SimpleNet(nn.Module):
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
# 3. LLC CALCULATION LOGIC
# ==========================================
def get_llc(model, loader, device, criterion, steps=100, burn_in=20):
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
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
            if i > burn_in: sampled_losses.append(loss.item())
    return (n * beta / 2) * (np.mean(sampled_losses) - w0_loss)

# ==========================================
# 4. TRAINING & EVALUATION LOOP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Subset for experiment
indices = torch.randperm(len(trainset))[:DATA_SIZE]
train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, indices), batch_size=64, shuffle=True)

model = SimpleNet(hidden=HIDDEN_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

history = {'epoch': [], 'loss': [], 'llc': []}

print(f"Starting experiment with DATA_SIZE={DATA_SIZE}...")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        epoch_loss += loss.item()
    
    if epoch % LLC_INTERVAL == 0:
        llc_val = get_llc(model, train_loader, device, criterion)
        history['epoch'].append(epoch)
        history['loss'].append(epoch_loss / len(train_loader))
        history['llc'].append(llc_val)
        print(f"Epoch {epoch}: Loss {history['loss'][-1]:.4f}, LLC {llc_val:.2f}")

# ==========================================
# 5. VISUALIZATION
# ==========================================
fig, ax1 = plt.subplots()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:red')
ax1.plot(history['epoch'], history['loss'], color='tab:red', label='Loss')
ax2 = ax1.twinx()
ax2.set_ylabel('LLC', color='tab:blue')
ax2.plot(history['epoch'], history['llc'], color='tab:blue', label='LLC')
plt.title(f'Phase Transition Experiment (Data Size: {DATA_SIZE})')
plt.show()