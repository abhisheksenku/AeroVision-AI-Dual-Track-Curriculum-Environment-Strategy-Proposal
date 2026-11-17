"""
=============================================================================
AeroVision AI - Phase 2: Deep Learning with PyTorch
Module 07: Custom Datasets, DataLoaders & Full Training Loop
=============================================================================

Domain Problem:
Aerial Land-Use & Asset Classification for Drone Surveys:
Classifying 32x32 aerial surface crops captured during autonomous drone flight:
- Class 0: 'SOLAR_PANEL'  (High-frequency grid texture)
- Class 1: 'VEGETATION'   (High green-channel variance)
- Class 2: 'WATER_BODY'   (Uniform high blue-channel distribution)
- Class 3: 'ASPHALT_ROAD' (Uniform neutral gray with low variance)

Learning Objectives:
1. Subclass `torch.utils.data.Dataset` implementing `__len__` and `__getitem__`.
2. PyTorch `DataLoader` mechanics: batching, shuffling, drop_last, and memory pinning.
3. Building a lightweight Convolutional Classifier (DroneLandClassifier).
4. Training vs Evaluation Modes: Why `model.train()` and `model.eval()` are critical for BatchNorm/Dropout.
5. Tracking training loss, validation loss, and validation accuracy.
6. Early Stopping mechanism: Tracking best validation loss and saving model weights.
7. Connection to AeroVision AI: Pre-filtering drone crops before detailed detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

# =============================================================================
# 1. SYNTHETIC AERIAL IMAGE DATASET
# =============================================================================
class SyntheticDroneAerialDataset(Dataset):
    """
    Generates synthetic 32x32x3 aerial image patches with distinct visual signatures.
    Outputs tensors in PyTorch format: (Channels=3, Height=32, Width=32) normalized to [0, 1].
    """
    def __init__(self, n_samples_per_class=150, seed=42):
        torch.manual_seed(seed)
        self.samples = []
        self.labels = []
        
        for _ in range(n_samples_per_class):
            # Class 0: SOLAR_PANEL (Grid pattern on dark blue/black background)
            panel = torch.zeros(3, 32, 32)
            panel[2, :, :] = 0.5  # Blue tint
            panel[:, ::4, :] += 0.4 # Grid lines
            panel[:, :, ::4] += 0.4
            panel += torch.randn(3, 32, 32) * 0.05
            self.samples.append(torch.clamp(panel, 0.0, 1.0))
            self.labels.append(0)
            
            # Class 1: VEGETATION (Dominated by green channel)
            veg = torch.zeros(3, 32, 32)
            veg[0, :, :] = torch.rand(32, 32) * 0.2  # Low red
            veg[1, :, :] = 0.5 + torch.rand(32, 32) * 0.4 # High green
            veg[2, :, :] = torch.rand(32, 32) * 0.2  # Low blue
            self.samples.append(torch.clamp(veg, 0.0, 1.0))
            self.labels.append(1)
            
            # Class 2: WATER_BODY (Smooth high blue, minimal variance)
            water = torch.zeros(3, 32, 32)
            water[0, :, :] = 0.1 + torch.randn(32, 32) * 0.02
            water[1, :, :] = 0.3 + torch.randn(32, 32) * 0.02
            water[2, :, :] = 0.7 + torch.randn(32, 32) * 0.03
            self.samples.append(torch.clamp(water, 0.0, 1.0))
            self.labels.append(2)
            
            # Class 3: ASPHALT_ROAD (Neutral dark gray)
            asphalt = 0.3 + torch.randn(3, 32, 32) * 0.04
            self.samples.append(torch.clamp(asphalt, 0.0, 1.0))
            self.labels.append(3)
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# =============================================================================
# 2. CONVOLUTIONAL CLASSIFIER ARCHITECTURE
# =============================================================================
class DroneLandClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Conv block 1: 32x32 -> 16x16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv block 2: 16x16 -> 8x8
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling: 8x8 -> 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.silu(self.bn1(self.conv1(x))))
        x = self.pool2(F.silu(self.bn2(self.conv2(x))))
        x = self.gap(x)
        x = torch.flatten(x, 1)  # Flatten from (B, 32, 1, 1) to (B, 32)
        logits = self.fc(x)
        return logits

# =============================================================================
# 3. TRAINING & VALIDATION PIPELINE
# =============================================================================
def main():
    section_header("1. SYNTHETIC AERIAL DATASET & DATALOADERS")
    full_dataset = SyntheticDroneAerialDataset(n_samples_per_class=150, seed=42)
    print(f"Total dataset size: {len(full_dataset)} aerial image patches")
    
    # 80/20 Train/Validation Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False)
    
    print(f"Train batches: {len(train_loader)} (32 samples/batch) | Val batches: {len(val_loader)}")
    
    # Inspect first batch
    sample_images, sample_labels = next(iter(train_loader))
    print(f"Batch Tensor Shape: Images={sample_images.shape}, Labels={sample_labels.shape}")

    # -------------------------------------------------------------------------
    section_header("2. MODEL INITIALIZATION & TRAINING LOOP")
    model = DroneLandClassifier(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print(f"{'Epoch':<8} | {'Train Loss':<14} | {'Val Loss':<14} | {'Val Accuracy':<15} | {'Status'}")
    print("-" * 65)

    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, 11):
        # ----------------- TRAIN PHASE -----------------
        model.train()  # Enables Dropout & sets BatchNorm to update running stats
        total_train_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)
            
        epoch_train_loss = total_train_loss / train_size
        
        # ----------------- EVAL PHASE ------------------
        model.eval()   # Disables Dropout & fixes BatchNorm stats
        total_val_loss = 0.0
        correct = 0
        
        with torch.no_grad():  # Disables Autograd engine for inference efficiency
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += torch.sum(preds == labels).item()
                
        epoch_val_loss = total_val_loss / val_size
        val_acc = (correct / val_size) * 100
        
        status = "Optimal"
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = model.state_dict()
            status = "Best Weight Saved"
            
        print(f"Epoch {epoch:<2} | {epoch_train_loss:<14.4f} | {epoch_val_loss:<14.4f} | {val_acc:>11.2f}%    | {status}")

    # Load best weights
    model.load_state_dict(best_weights)
    print(f"\n--> Training Complete! Model converged with Best Validation Loss: {best_val_loss:.4f}")

    # -------------------------------------------------------------------------
    section_header("3. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: What physically happens inside PyTorch when you switch between `model.train()` and `model.eval()`?
A1: Certain layers behave completely differently during training versus inference:
    1. `nn.Dropout`:
       - `train()`: Randomly zeros activations with probability p and scales remaining elements by 1/(1-p).
       - `eval()`: Disables all zeroing; passes activations through as an identity operation.
    2. `nn.BatchNorm2d`:
       - `train()`: Computes batch mean and batch variance directly from the current mini-batch, and updates
         its persistent running mean and running variance buffers using an exponential moving average.
       - `eval()`: Freezes the running statistics; normalizes inputs strictly using the pre-computed population
         statistics (`running_mean` and `running_var`), ensuring deterministic outputs regardless of batch size.
    Forgetting to call `model.eval()` during deployment causes predictions on single images (batch size 1)
    to fail because batch variance cannot be computed from a single sample!

Q2: What is the purpose of `torch.nn.AdaptiveAvgPool2d((1, 1))` (Global Average Pooling)?
A2: In classical architectures (AlexNet, VGG), convolutional feature maps were flattened into massive linear layers
    (e.g., 7x7x512 = 25,088 inputs), causing parameter explosion (100M+ weights) and extreme susceptibility to overfitting.
    Global Average Pooling averages each feature channel across its entire spatial height and width down to a single number (1x1).
    Benefits:
    1. Slashing parameters: The classifier layer only needs (C_channels * Num_Classes) weights instead of (H * W * C * Num_Classes).
    2. Arbitrary Input Resolution: The network can accept images of any input resolution without changing classifier layer dimensions!
""")

if __name__ == "__main__":
    main()
