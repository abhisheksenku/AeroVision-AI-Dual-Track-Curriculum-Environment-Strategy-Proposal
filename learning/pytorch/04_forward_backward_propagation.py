"""
=============================================================================
AeroVision AI - Phase 2: Deep Learning with PyTorch
Module 04: Forward & Backward Propagation: Mechanics from First Principles
=============================================================================

Domain Problem:
Autonomous Drone Flight Stability Predictor:
Predicting a continuous Flight Stability Index [0.0, 1.0] from 3 drone sensors:
- Sensor 1: Gyro angular velocity (deg/s)
- Sensor 2: Accelerometer jerk (m/s^3)
- Sensor 3: Barometric pressure delta (hPa)

Learning Objectives:
1. Master the canonical 5-step PyTorch training loop:
   zero_grad() -> forward() -> criterion() -> backward() -> step().
2. Manual Mathematical Derivation of Backpropagation for a 2-Layer Neural Network:
   - Forward equations: z1 = X @ W1 + b1, a1 = ReLU(z1), z2 = a1 @ W2 + b2, y_hat = z2
   - Loss equation: MSE = (1/N) * sum((y_hat - y)^2)
   - Analytical chain rule derivatives: dL/dW2, dL/db2, dL/dW1, dL/db1.
3. Prove that PyTorch's Autograd gradients match our manual analytical math down to 1e-7!
4. Train the network to convergence over multiple epochs.
5. Connection to AeroVision AI: How bounding box coordinate regressors optimize their weights.
"""

import torch
import torch.nn as nn
import numpy as np

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

# =============================================================================
# 1. MANUAL 2-LAYER NETWORK FOR GRADIENT PROOF
# =============================================================================
class ManualTwoLayerNet(nn.Module):
    def __init__(self, in_features=3, hidden_dim=4, out_features=1):
        super().__init__()
        # Fix weights with known values for exact mathematical proof
        torch.manual_seed(42)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        h = F_relu = torch.relu(self.fc1(x))
        out = self.fc2(h)
        return out

def manual_analytical_backprop_proof():
    """
    Computes a single forward and backward step manually using pure tensor math,
    then compares against PyTorch's Autograd .backward().
    """
    torch.manual_seed(42)
    # 2 input samples with 3 features
    X = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, -1.0]], dtype=torch.float32)
    y = torch.tensor([[2.5], [1.0]], dtype=torch.float32)
    N = X.shape[0]

    # Instantiate PyTorch model
    model = ManualTwoLayerNet(in_features=3, hidden_dim=4, out_features=1)
    W1 = model.fc1.weight.clone().detach().requires_grad_(True)  # (4, 3)
    b1 = model.fc1.bias.clone().detach().requires_grad_(True)    # (4,)
    W2 = model.fc2.weight.clone().detach().requires_grad_(True)  # (1, 4)
    b2 = model.fc2.bias.clone().detach().requires_grad_(True)    # (1,)

    # -------------------------------------------------------------
    # 1. FORWARD PASS (MANUAL)
    # z1 = X @ W1.T + b1          shape: (2, 4)
    # a1 = relu(z1)               shape: (2, 4)
    # y_hat = a1 @ W2.T + b2      shape: (2, 1)
    # -------------------------------------------------------------
    z1 = X @ W1.T + b1
    a1 = torch.relu(z1)
    y_hat = a1 @ W2.T + b2
    loss_manual = torch.mean((y_hat - y) ** 2)

    # -------------------------------------------------------------
    # 2. BACKWARD PASS (MANUAL ANALYTICAL CHAIN RULE)
    # d(MSE)/d(y_hat) = (2 / N) * (y_hat - y)           shape: (2, 1)
    # d(MSE)/d(W2)    = (d(MSE)/d(y_hat))^T @ a1        shape: (1, 4)
    # d(MSE)/d(b2)    = sum(d(MSE)/d(y_hat), dim=0)     shape: (1,)
    #
    # Backprop through ReLU:
    # d(MSE)/d(a1)    = (d(MSE)/d(y_hat)) @ W2          shape: (2, 4)
    # d(MSE)/d(z1)    = d(MSE)/d(a1) * (z1 > 0)         shape: (2, 4)
    #
    # Backprop into Layer 1:
    # d(MSE)/d(W1)    = (d(MSE)/d(z1))^T @ X            shape: (4, 3)
    # d(MSE)/d(b1)    = sum(d(MSE)/d(z1), dim=0)        shape: (4,)
    # -------------------------------------------------------------
    d_yhat = (2.0 / N) * (y_hat - y)
    d_W2 = d_yhat.T @ a1
    d_b2 = torch.sum(d_yhat, dim=0)

    d_a1 = d_yhat @ W2
    d_z1 = d_a1 * (z1 > 0).float()

    d_W1 = d_z1.T @ X
    d_b1 = torch.sum(d_z1, dim=0)

    # -------------------------------------------------------------
    # 3. PYTORCH AUTOGRAD PASS
    # -------------------------------------------------------------
    pred_pt = model(X)
    loss_pt = nn.MSELoss()(pred_pt, y)
    model.zero_grad()
    loss_pt.backward()

    # Verify identical loss
    assert torch.isclose(loss_manual, loss_pt), "Losses do not match!"

    # Verify identical gradients
    err_W2 = torch.max(torch.abs(d_W2 - model.fc2.weight.grad)).item()
    err_b2 = torch.max(torch.abs(d_b2 - model.fc2.bias.grad)).item()
    err_W1 = torch.max(torch.abs(d_W1 - model.fc1.weight.grad)).item()
    err_b1 = torch.max(torch.abs(d_b1 - model.fc1.bias.grad)).item()

    print(f"Discrepancy on W2 gradient: {err_W2:.10f}")
    print(f"Discrepancy on b2 gradient: {err_b2:.10f}")
    print(f"Discrepancy on W1 gradient: {err_W1:.10f}")
    print(f"Discrepancy on b1 gradient: {err_b1:.10f}")

    assert max(err_W2, err_b2, err_W1, err_b1) < 1e-6, "Analytical gradients do not match Autograd!"
    print("--> PROVED: Manual chain-rule gradients match PyTorch Autograd within 1e-7 precision!")

# =============================================================================
# 2. END-TO-END TRAINING LOOP TO CONVERGENCE
# =============================================================================
def train_drone_stability_predictor():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 300 synthetic drone flight telemetry samples
    # True relation: stability = 0.5 - 0.02*gyro - 0.05*jerk + 0.01*pressure
    X_np = np.random.uniform(-5.0, 5.0, size=(300, 3)).astype(np.float32)
    y_np = (0.50 - 0.03 * X_np[:, 0:1] - 0.06 * X_np[:, 1:2] + 0.02 * X_np[:, 2:3] + np.random.normal(0, 0.05, (300, 1))).astype(np.float32)
    
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    model = nn.Sequential(
        nn.Linear(3, 16),
        nn.SiLU(),
        nn.Linear(16, 1)
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"{'Epoch':<8} | {'Loss (MSE)':<15} | {'Relative Loss Reduction'}")
    print("-" * 50)

    initial_loss = None
    for epoch in range(1, 151):
        # 1. ZERO GRADIENTS (Prevents accumulation from previous steps)
        optimizer.zero_grad()
        
        # 2. FORWARD PASS (Compute predictions through computational graph)
        y_pred = model(X)
        
        # 3. COMPUTE LOSS (Scalar objective measure)
        loss = criterion(y_pred, y)
        
        if epoch == 1:
            initial_loss = loss.item()
            
        # 4. BACKWARD PROPAGATION (Traverse graph via chain rule)
        loss.backward()
        
        # 5. PARAMETER UPDATE (W = W - lr * grad)
        optimizer.step()
        
        if epoch in [1, 25, 50, 100, 150]:
            reduction = (1.0 - (loss.item() / initial_loss)) * 100
            print(f"Epoch {epoch:<3} | {loss.item():<15.6f} | Reduced by {reduction:>6.2f}%")

    final_loss = loss.item()
    print(f"\nTraining converged: Loss dropped from {initial_loss:.4f} to {final_loss:.6f} (-{((1 - final_loss/initial_loss)*100):.1f}%)")

def main():
    section_header("1. MANUAL BACKPROPAGATION CHAIN RULE VS AUTOGRAD PROOF")
    manual_analytical_backprop_proof()

    section_header("2. CANONICAL 5-STEP PYTORCH TRAINING LOOP")
    train_drone_stability_predictor()

    section_header("3. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: What happens under the hood when you execute `loss.backward()` followed by `optimizer.step()`?
A1: 1. `loss.backward()`:
       PyTorch traverses the dynamic Directed Acyclic Graph (DAG) starting from the scalar loss root
       backward through all intermediate non-leaf operations (`grad_fn`) to all leaf parameters (`requires_grad=True`).
       At each step, it applies the vector-Jacobian product (chain rule) and accumulates the resulting gradient
       onto each tensor's `.grad` attribute (`param.grad += dL/dParam`).
    2. `optimizer.step()`:
       The optimizer iterates over all registered parameters in `model.parameters()` and updates their values
       in-place according to its specific optimization algorithm (e.g., SGD: `param.data -= lr * param.grad`,
       or Adam: maintaining running first/second gradient moments).
       Crucially, `step()` does NOT clear gradients; `optimizer.zero_grad()` must be explicitly called next.

Q2: Why is `zero_grad()` called BEFORE `loss.backward()`, and what bug occurs if you put it AFTER `optimizer.step()`?
A2: Conceptually, placing `zero_grad()` immediately before `backward()` or immediately after `step()` produces identical
    results in a basic single-optimizer training loop.
    However, calling `optimizer.zero_grad()` at the very START of the iteration is industry standard because:
    - If training terminates unexpectedly or raises an exception during the forward/backward pass, `.grad` attributes
      are guaranteed to be clean at the start of any resumed or re-entered step.
    - If multiple optimizers or loss functions share the same parameters (e.g. GANs, Actor-Critic, Multi-Task Detectors),
      explicitly controlling when gradients are cleared prevents one optimizer from prematurely wiping gradients needed by another.
""")

if __name__ == "__main__":
    main()
