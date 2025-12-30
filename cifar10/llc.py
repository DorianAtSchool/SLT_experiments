import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# --- 1. THE MODEL ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- 2. THE SGLD OPTIMIZER ---
class SGLD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=1.0):
        defaults = dict(lr=lr, beta=beta)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            # Noise scale: sqrt(2 * lr / beta)
            noise_std = np.sqrt(2 * group['lr'] / group['beta'])
            for p in group['params']:
                if p.grad is None: continue
                noise = torch.randn_like(p.data) * noise_std
                p.data.add_(p.grad.data, alpha=-group['lr'])
                p.data.add_(noise)

# --- 3. LLC CALCULATION FUNCTION ---
def calculate_llc(model, device, loader, criterion, lr=1e-4, steps=200, burn_in=50):
    n = len(loader.dataset)
    beta = 1 / np.log(n)
    
    # Baseline Loss
    model.eval()
    with torch.no_grad():
        w0_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in loader) / len(loader)

    # Sampling
    optimizer = SGLD(model.parameters(), lr=lr, beta=beta)
    sampled_losses = []
    model.train()
    
    for i in range(steps):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            if i > burn_in:
                sampled_losses.append(loss.item())
                
    avg_loss = np.mean(sampled_losses)
    llc = (n * beta / 2) * (avg_loss - w0_loss)
    return llc

# --- 4. MAIN EXECUTION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Use a subset for faster LLC calculation
subset_indices = range(1000)
subset_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, subset_indices), batch_size=64)
full_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(5): # Train for 5 epochs
    model.train()
    for x, y in full_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); criterion(model(x), y).backward(); optimizer.step()
    
    # Calculate LLC at the end of each epoch
    current_llc = calculate_llc(model, device, subset_loader, criterion)
    print(f"Epoch {epoch+1} Complete. LLC: {current_llc:.4f}")