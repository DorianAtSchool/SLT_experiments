import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP: MODULAR ARITHMETIC TASK
# ==========================================
P = 99  # Prime modulus (common in SLT papers)
TRAIN_FRACTION = 0.5 # Use only 50% of data to force the model to 'grok' the rest
NUM_EPOCHS = 2000000
LLC_INTERVAL =100

# Create all possible pairs (a, b) and targets (a+b)%P
pairs = torch.tensor([(i, j) for i in range(P) for j in range(P)])
targets = (pairs[:, 0] + pairs[:, 1]) % P

# Split into train/val
indices = torch.randperm(P * P)
split = int(P * P * TRAIN_FRACTION)
train_idx, val_idx = indices[:split], indices[split:]

# ==========================================
# 2. MODEL: 2-LAYER MLP
# ==========================================
class SLT_Net(nn.Module):
    def __init__(self, p, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(p, hidden) 
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, p)
        )
    def forward(self, x):
        e = self.embed(x).view(x.shape[0], -1) 
        return self.fc(e)

# ==========================================
# 3. SGLD OPTIMIZER & LLC CALCULATION
# ==========================================
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

def calculate_llc(model, idx_list, criterion, steps=100):
    n = len(idx_list)
    beta = 1 / np.log(n)
    x, y = pairs[idx_list], targets[idx_list]
    model.eval()
    with torch.no_grad(): 
        w0_loss = criterion(model(x), y).item()
    optimizer = SGLD(model.parameters(), lr=1e-5, beta=beta)
    losses = []
    model.train()
    for i in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y); loss.backward(); optimizer.step()
        if i > 20: losses.append(loss.item())
    return (n * beta / 2) * (np.mean(losses) - w0_loss)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SLT_Net(P).to(device)
pairs, targets = pairs.to(device), targets.to(device)

# High weight decay (0.1) is the 'pressure' that resolves the singularity
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1)
criterion = nn.CrossEntropyLoss()

history = {'epoch': [], 't_loss': [], 't_acc': [], 'v_acc': [], 'llc': []}

print(f"Starting Modular Addition Grokking Experiment (P={P})...")

for epoch in range(NUM_EPOCHS):
    model.train()
    # Train on the subset
    optimizer.zero_grad()
    loss = criterion(model(pairs[train_idx]), targets[train_idx])
    loss.backward()
    optimizer.step()
    
    if epoch % LLC_INTERVAL == 0:
        model.eval()
        with torch.no_grad():
            # Training Metrics
            t_logits = model(pairs[train_idx])
            tl = criterion(t_logits, targets[train_idx]).item()
            ta = (t_logits.argmax(1) == targets[train_idx]).float().mean().item() * 100
            
            # Validation Metrics
            v_logits = model(pairs[val_idx])
            va = (v_logits.argmax(1) == targets[val_idx]).float().mean().item() * 100
            
        llc = calculate_llc(model, train_idx, criterion)
        
        history['epoch'].append(epoch); history['t_loss'].append(tl)
        history['t_acc'].append(ta); history['v_acc'].append(va); history['llc'].append(llc)
        
        print(f"Ep {epoch:04d} | T-Loss: {tl:.4f} | T-Acc: {ta:5.1f}% | V-Acc: {va:5.1f}% | LLC: {llc:6.2f}")

# ==========================================
# 5. VISUALIZATION
# ==========================================
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Accuracies
ax1.plot(history['epoch'], history['t_acc'], 'g--', label='Train Accuracy')
ax1.plot(history['epoch'], history['v_acc'], 'b-', linewidth=2, label='Val Accuracy (Grokking)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(-5, 105)

# Plot LLC on secondary Y-axis
ax2 = ax1.twinx()
ax2.plot(history['epoch'], history['llc'], 'r-o', alpha=0.4, markersize=4, label='LLC (Complexity)')
ax2.set_ylabel('Local Learning Coefficient (LLC)')

plt.title(f"Exact SLT Replication: Grokking modular addition (P={P})")
ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.grid(True, alpha=0.2)
plt.show()