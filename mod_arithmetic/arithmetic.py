"""
An attempt to recreate the Power et al. (2022) experiment on modular arithmetic
Grokking: Generalization Beyond Overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. Dataset Generation
# ============================================================================

class ModularArithmeticDataset(Dataset):
    """Dataset for binary operations like a + b = c (mod p)"""
    
    def __init__(self, p, operation='add', train_fraction=0.5, train=True):
        """
        Args:
            p: Prime modulus
            operation: 'add', 'subtract', 'divide', or 'multiply'
            train_fraction: Fraction of data for training
            train: Whether this is training or validation set
        """
        self.p = p
        self.operation = operation
        
        # Generate all possible equations
        equations = []
        if operation == 'divide':
            # For division, b cannot be 0
            for a in range(p):
                for b in range(1, p):  # b from 1 to p-1
                    c = (a * pow(b, -1, p)) % p  # modular division
                    equations.append((a, b, c))
        else:
            for a in range(p):
                for b in range(p):
                    if operation == 'add':
                        c = (a + b) % p
                    elif operation == 'subtract':
                        c = (a - b) % p
                    elif operation == 'multiply':
                        c = (a * b) % p
                    equations.append((a, b, c))
        
        # Shuffle and split
        np.random.shuffle(equations)
        split_idx = int(len(equations) * train_fraction)
        
        if train:
            self.data = equations[:split_idx]
        else:
            self.data = equations[split_idx:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        a, b, c = self.data[idx]
        # Input: [a, op_token, b, equals_token]
        # We'll use p for op token and p+1 for equals token
        input_seq = torch.tensor([a, self.p, b, self.p + 1], dtype=torch.long)
        target = torch.tensor(c, dtype=torch.long)
        return input_seq, target


# ============================================================================
# 2. Transformer Model
# ============================================================================

class PositionalEncoding(nn.Module):
    """Standard positional encoding"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DecoderTransformer(nn.Module):
    """Decoder-only transformer as used in the paper"""
    
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection (only for predicting the answer)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # x shape: (batch, seq_len)
        batch_size, seq_len = x.shape
        
        # Embed and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Create causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Pass through transformer
        # For decoder, we need to pass the same thing as memory
        output = self.transformer(x, x, tgt_mask=mask, memory_mask=mask)
        
        # Only return the last position (where answer should be)
        output = output[:, -1, :]  # (batch, d_model)
        
        # Project to vocabulary
        logits = self.output_proj(output)  # (batch, vocab_size)
        
        return logits


# ============================================================================
# 3. Training Loop
# ============================================================================

def train_epoch(model, train_loader, optimizer, device, p):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    
    return total_loss / len(train_loader), correct / total


def evaluate(model, val_loader, device, p):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    
    return total_loss / len(val_loader), correct / total


# ============================================================================
# 4. Main Experiment
# ============================================================================

def run_grokking_experiment(
    p=97,
    operation='divide',
    train_fraction=0.5,
    n_epochs=50000,
    batch_size=512,
    lr=1e-3,
    weight_decay=1.0,
    log_interval=100
):
    """
    Run the grokking experiment
    
    Args:
        p: Prime modulus (default 97 as in paper)
        operation: Type of operation ('add', 'subtract', 'divide', 'multiply')
        train_fraction: Fraction of data for training
        n_epochs: Number of training epochs (paper used up to 1M steps)
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay coefficient
        log_interval: How often to log metrics
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print(f"\nGenerating datasets for {operation} mod {p}...")
    train_dataset = ModularArithmeticDataset(p, operation, train_fraction, train=True)
    val_dataset = ModularArithmeticDataset(p, operation, train_fraction, train=False)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create dataloaders
    actual_batch_size = min(batch_size, len(train_dataset) // 2)
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False)
    
    # Create model
    vocab_size = p + 2  # p numbers + op_token + equals_token
    model = DecoderTransformer(vocab_size=vocab_size).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer (AdamW with weight decay as in paper)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        weight_decay=weight_decay
    )
    
    # Learning rate warmup
    def lr_lambda(epoch):
        if epoch < 10:
            return epoch / 10
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': []
    }
    
    print("\nStarting training...")
    print("=" * 70)
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, p)
        scheduler.step()
        
        # Evaluate
        if epoch % log_interval == 0 or epoch == n_epochs - 1:
            val_loss, val_acc = evaluate(model, val_loader, device, p)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['epochs'].append(epoch)
            
            if epoch % (log_interval * 10) == 0:
                print(f"\nEpoch {epoch:5d} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping if we achieve perfect generalization
        # if epoch % log_interval == 0:
        #     val_loss, val_acc = evaluate(model, val_loader, device, p)
        #     if train_acc > 0.99 and val_acc > 0.99:
        #         print(f"\n✓ Perfect generalization achieved at epoch {epoch}!")
        #         history['train_loss'].append(train_loss)
        #         history['train_acc'].append(train_acc)
        #         history['val_loss'].append(val_loss)
        #         history['val_acc'].append(val_acc)
        #         history['epochs'].append(epoch)
        #         break
    
    return history, model


def plot_grokking(history):
    """Plot the grokking phenomenon"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = history['epochs']
    
    # Plot accuracy
    ax1.plot(epochs, history['train_acc'], 'r-', label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Grokking: Accuracy Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot loss
    ax2.plot(epochs, history['train_loss'], 'r-', label='Train Loss', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'g-', label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Over Time (Double Descent)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 5. Run the experiment!
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GROKKING EXPERIMENT: Recreation of Power et al. (2022)")
    print("=" * 70)
    
    # Run experiment with settings from the paper
    history, model = run_grokking_experiment(
        p=97,                    # Prime modulus
        operation='divide',      # Division mod 97 (shows strongest grokking)
        train_fraction=0.5,      # 50% training data
        n_epochs=1000,          # Adjust based on compute (paper used up to 1M steps)
        batch_size=512,
        lr=1e-3,
        weight_decay=1.0,        # Critical for grokking!
        log_interval=100
    )
    
    # Plot results
    print("\nPlotting results...")
    plot_grokking(history)
    
    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)
    print("\nKey observations:")
    print("- Training accuracy should reach ~100% relatively quickly")
    print("- Validation accuracy should lag behind significantly (grokking!)")
    print("- If you see the gap close, you've witnessed grokking!")
    print("\nTry adjusting:")
    print("- train_fraction: smaller values show stronger grokking")
    print("- weight_decay: set to 0 to prevent grokking!")
    print("- operation: 'add', 'subtract', 'multiply', or 'divide'")