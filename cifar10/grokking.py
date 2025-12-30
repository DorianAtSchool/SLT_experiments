import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION (The "Complexity" Knobs)
# ==========================================
DATA_SIZE = 1000       # Total training images (Smaller = easier to see Grokking)
VAL_SIZE = 500        # Images to track generalization
NUM_EPOCHS = 100
LLC_INTERVAL = 5      # How often to compute LLC (expensive)
SGLD_STEPS = 100      # Steps for LLC estimation
BURN_IN = 20          # Burn-in for SGLD

# ==========================================
# 2. MODEL ARCHITECTURE (Simple MLP)
# ==========================================
class GrokkingNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10)
        )
    def forward(self, x): 
        return self.main(x)

# ==========================================
# 3. SGLD OPTIMIZER FOR LLC
# ==========================================
class SGLD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=1.0):
        super().__init__(params, dict(lr=lr, beta=beta))
    def step(self):
        for group in self.param_groups:
            # Noise scale: sqrt(2 * lr / beta)
            noise_std = np.sqrt(2 * group['lr'] / group['beta'])
            for p in group['params']:
                if p.grad is None: continue
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(p.grad.data, alpha=-group['lr'])
                p.data.add_(noise)

# ==========================================
# 4. LLC & ACCURACY EVALUATION FUNCTIONS
# ==========================================
def get_llc(model, loader, device, criterion, steps=SGLD_STEPS, burn_in=BURN_IN):
    n = len(loader.dataset)
    beta = 1 / np.log(n)
    model.eval()
    
    # Calculate baseline loss at current weights
    with torch.no_grad():
        w0_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in loader) / len(loader)
    
    # Sample neighborhood with SGLD
    optimizer = SGLD(model.parameters(), lr=1e-5, beta=beta)
    sampled_losses = []
    model.train() # Enable gradients
    for i in range(steps):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            if i > burn_in: 
                sampled_losses.append(loss.item())
    
    # Formula: LLC = (n * beta / 2) * (Avg_Loss - Base_Loss)
    return (n * beta / 2) * (np.mean(sampled_losses) - w0_loss)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total

# ==========================================
# 5. DATA PREPARATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
full_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create subsets to induce Grokking (Grokking happens on small datasets)
train_indices = torch.randperm(len(full_train))[:DATA_SIZE]
test_indices = torch.randperm(len(full_test))[:VAL_SIZE]

train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_train, train_indices), batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_test, test_indices), batch_size=64)

# ==========================================
# 6. TRAINING LOOP WITH LLC TRACKING
# ==========================================
model = GrokkingNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2) # Weight decay helps grokking
criterion = nn.CrossEntropyLoss()

history = {'epoch': [], 'train_acc': [], 'val_acc': [], 'llc': []}

print(f"Training on {DATA_SIZE} images. Tracking LLC every {LLC_INTERVAL} epochs...")

for epoch in range(NUM_EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    
    if epoch % LLC_INTERVAL == 0:
        t_acc = evaluate(model, train_loader, device)
        v_acc = evaluate(model, val_loader, device)
        llc_val = get_llc(model, train_loader, device, criterion)
        
        history['epoch'].append(epoch)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['llc'].append(llc_val)
        
        print(f"Epoch {epoch:03d} | Train Acc: {t_acc:3.1f}% | Val Acc: {v_acc:3.1f}% | LLC: {llc_val:6.2f}")

# ==========================================
# 7. VISUALIZATION OF THE PHASE TRANSITION
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Accuracies
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy (%)')
ax1.plot(history['epoch'], history['train_acc'], label='Train Accuracy', color='green', linestyle='--')
ax1.plot(history['epoch'], history['val_acc'], label='Validation Accuracy', color='blue', linewidth=2)
ax1.legend(loc='upper left')

# Plot LLC on second Y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('LLC (Complexity)')
ax2.plot(history['epoch'], history['llc'], label='Local Learning Coeff (LLC)', color='red', marker='o')
ax2.legend(loc='upper right')

plt.title(f'LLC and Grokking Phase Transition\n(Data: {DATA_SIZE} samples, Model: MLP)')
plt.grid(True, alpha=0.3)
plt.show()