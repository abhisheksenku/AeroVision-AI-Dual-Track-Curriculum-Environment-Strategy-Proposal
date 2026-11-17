"""
=============================================================================
AeroVision AI - Phase 2: Deep Learning with PyTorch
Module 05: Loss Functions, Modern Optimizers & AdamW Mechanics
=============================================================================

Domain Problem:
Optimizing a Multi-Task Drone Detection Head:
In aerial vision systems, gradients from small distant objects (e.g., pedestrians,
bicycles) are often 100x smaller than gradients from large structures.
Selecting the right loss function (Smooth L1 vs BCEWithLogits) and optimizer (SGD with
Momentum vs AdamW) directly determines whether the detector converges or collapses.

Learning Objectives:
1. Deep dive into Core PyTorch Loss Functions:
   - nn.MSELoss vs nn.SmoothL1Loss (Huber): Outlier robustness for coordinates.
   - nn.BCEWithLogitsLoss: The log-sum-exp numerical stability trick for multi-label classification.
   - nn.CrossEntropyLoss: Softmax + NLLLoss for multi-class identification.
2. Mathematical Formulation of Optimizers:
   - SGD with Momentum: Overcoming ravines and damping high-frequency oscillations.
   - Adam: Combining first moment (momentum) and second moment (adaptive variance).
   - AdamW: Why L2 regularization in standard Adam is mathematically flawed, and how
     decoupled weight decay restores true L2 shrinkage.
3. Learning Rate Scheduling: Cosine Annealing (torch.optim.lr_scheduler.CosineAnnealingLR)
   as used in Ultralytics YOLO11.
4. Empirical Optimizer Race: Benchmarking SGD vs Momentum vs Adam vs AdamW on an ill-conditioned surface.
5. Connection to AeroVision AI: Choosing optimizers for drone vision fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

# =============================================================================
# 1. LOSS FUNCTIONS IN COMPUTER VISION
# =============================================================================
def compare_loss_functions():
    section_header("1. REGRESSION & CLASSIFICATION LOSS FUNCTIONS")
    
    # -------------------------------------------------------------
    # A. MSE vs Smooth L1 (Huber) for Bounding Boxes
    # -------------------------------------------------------------
    pred_box = torch.tensor([10.0, 50.0], requires_grad=True)
    # Target 1: Small error (0.5 px) | Target 2: Massive outlier error (40.0 px) due to drone camera vibration
    target_box = torch.tensor([10.5, 10.0])
    
    mse_loss = nn.MSELoss()(pred_box, target_box)
    huber_loss = nn.SmoothL1Loss(beta=1.0)(pred_box, target_box)
    
    # Gradients of MSE vs Huber on the outlier (diff = 40.0)
    # d(MSE)/d(diff) = 2 * diff = 80.0 (Gradients explode!)
    # d(Huber)/d(diff) = 1.0 (Linear gradient clips outlier explosion!)
    mse_loss.backward(retain_graph=True)
    grad_mse = pred_box.grad.clone()
    pred_box.grad.zero_()
    
    huber_loss.backward()
    grad_huber = pred_box.grad.clone()
    
    print("Bounding Box Loss Comparison (Small error vs Outlier error [40.0 px]):")
    print(f"  MSE Loss Value:       {mse_loss.item():.2f} | Outlier Gradient: {grad_mse[1].item():.2f}")
    print(f"  Smooth L1 Loss Value: {huber_loss.item():.2f} | Outlier Gradient: {grad_huber[1].item():.2f}")
    print("  --> Smooth L1 clips outlier gradient to constant 1.0, preventing training destabilization.")

    # -------------------------------------------------------------
    # B. BCEWithLogitsLoss: Numerical Stability Trick
    # -------------------------------------------------------------
    # Naive: sigmoid(z) followed by log() underflows when z is a large negative number
    # BCEWithLogitsLoss mathematically simplifies:
    # - [y * log(sigmoid(z)) + (1-y) * log(1 - sigmoid(z))]
    # = max(z, 0) - z*y + log(1 + exp(-|z|))  -> Zero underflow / overflow risk!
    raw_logits = torch.tensor([-80.0, 50.0, 0.0])  # Extreme logits
    binary_targets = torch.tensor([0.0, 1.0, 1.0])
    
    bce_stable = nn.BCEWithLogitsLoss()(raw_logits, binary_targets)
    print(f"\nBCEWithLogitsLoss with extreme logits [-80, 50]: {bce_stable.item():.4f} (Clean & Finite!)")

# =============================================================================
# 2. ADAM VS ADAMW: DECOUPLED WEIGHT DECAY PROOF
# =============================================================================
def verify_adamw_mechanics():
    section_header("2. ADAM VS ADAMW: THE MATHEMATICAL DIFFERENCE")
    # Why standard Adam with weight decay fails:
    # In standard Adam with L2 penalty, the weight decay term (lambda * W) is added
    # directly to the gradient g_t:
    #   g_t = grad + lambda * W
    #   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    #   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    #   W = W - (alpha / sqrt(v_t)) * m_t
    # Notice that the weight decay term is DIVIDED by sqrt(v_t)!
    # If a parameter has historically large gradients (large v_t), its weight decay is
    # artificially suppressed! If it has small gradients (small v_t), its weight decay is amplified!
    #
    # In AdamW (Loshchilov & Hutter), weight decay is DECOUPLED:
    #   W = W - alpha * (lambda * W) - (alpha / sqrt(v_t)) * m_t
    # Weight decay operates at the exact constant rate alpha * lambda, restoring true L2 regularization!
    
    w_adam = nn.Parameter(torch.tensor([10.0]))
    w_adamw = nn.Parameter(torch.tensor([10.0]))
    
    opt_adam = torch.optim.Adam([w_adam], lr=0.1, weight_decay=0.1)
    opt_adamw = torch.optim.AdamW([w_adamw], lr=0.1, weight_decay=0.1)
    
    # Large gradient step (e.g. gradient = 50.0)
    for _ in range(5):
        # Fake loss with gradient 50.0
        loss_a = w_adam * 50.0
        opt_adam.zero_grad()
        loss_a.backward()
        opt_adam.step()
        
        loss_w = w_adamw * 50.0
        opt_adamw.zero_grad()
        loss_w.backward()
        opt_adamw.step()
        
    print(f"Weight after 5 steps under Adam (coupled decay):  {w_adam.item():.4f}")
    print(f"Weight after 5 steps under AdamW (decoupled decay):{w_adamw.item():.4f}")
    print("--> AdamW regularized the large parameter more aggressively because weight decay was not suppressed by large v_t!")

# =============================================================================
# 3. EMPIRICAL OPTIMIZER BENCHMARK ON NON-CONVEX ILL-CONDITIONED SURFACE
# =============================================================================
def run_optimizer_race():
    section_header("3. OPTIMIZER BENCHMARK: SGD VS MOMENTUM VS ADAM VS ADAMW")
    # Ill-conditioned anisotropic quadratic bowl (Rosenbrock-like ravine):
    # f(x, y) = 0.5 * x^2 + 25.0 * y^2
    # The surface has high curvature along y and low curvature along x (steep valley).
    def loss_surface(params):
        x, y = params[0], params[1]
        return 0.5 * (x ** 2) + 25.0 * (y ** 2)

    optimizers_dict = {
        "1. Pure SGD (lr=0.03)": lambda p: torch.optim.SGD(p, lr=0.03),
        "2. SGD + Momentum (0.9)": lambda p: torch.optim.SGD(p, lr=0.03, momentum=0.9),
        "3. RMSprop (lr=0.03)": lambda p: torch.optim.RMSprop(p, lr=0.03),
        "4. Adam (lr=0.03)": lambda p: torch.optim.Adam(p, lr=0.03),
        "5. AdamW (lr=0.03, wd=0.01)": lambda p: torch.optim.AdamW(p, lr=0.03, weight_decay=0.01)
    }

    print(f"{'Optimizer':<28} | {'Final Loss (after 50 steps)':<25} | {'Final Coordinates [x, y]'}")
    print("-" * 75)

    for name, opt_factory in optimizers_dict.items():
        # Start far up the valley wall at (x=10.0, y=10.0)
        p = nn.Parameter(torch.tensor([10.0, 10.0]))
        opt = opt_factory([p])
        
        for step in range(50):
            opt.zero_grad()
            l = loss_surface(p)
            l.backward()
            opt.step()
            
        final_l = loss_surface(p).item()
        coords_str = f"[{p[0].item():.2f}, {p[1].item():.2f}]"
        print(f"{name:<28} | {final_l:<25.6f} | {coords_str}")

# =============================================================================
# 4. LEARNING RATE SCHEDULER: COSINE ANNEALING
# =============================================================================
def test_cosine_scheduler():
    section_header("4. COSINE ANNEALING LR SCHEDULER (YOLO11 PIPELINE)")
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=0.01)
    # Cosine annealing over T_max=100 epochs, decaying to eta_min=1e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    
    recorded_lrs = []
    epochs_to_check = [0, 25, 50, 75, 100]
    
    for epoch in range(101):
        if epoch in epochs_to_check:
            current_lr = optimizer.param_groups[0]['lr']
            recorded_lrs.append((epoch, current_lr))
        optimizer.step()
        scheduler.step()
        
    print(f"{'Epoch':<10} | {'Learning Rate':<18} | {'Decay Percentage'}")
    print("-" * 50)
    base_lr = recorded_lrs[0][1]
    for ep, lr in recorded_lrs:
        pct = (1.0 - (lr / base_lr)) * 100
        print(f"Epoch {ep:<4} | {lr:<18.6f} | {pct:>6.2f}% decayed")

def main():
    compare_loss_functions()
    verify_adamw_mechanics()
    run_optimizer_race()
    test_cosine_scheduler()

    section_header("5. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: Why should you use `nn.BCEWithLogitsLoss` instead of `nn.Sigmoid()` followed by `nn.BCELoss()`?
A1: Mathematical Stability:
    When raw model output `z` is a large negative number (e.g. z = -80.0), `sigmoid(-80.0)` underflows to exactly 0.0 in float32.
    Passing 0.0 to `torch.log(0.0)` produces `-inf` or `NaN`, which poisons the entire gradient graph and destroys model weights.
    Conversely, when z is large positive (+80.0), `1 - sigmoid(80.0)` underflows to 0.0, similarly yielding `NaN`.
    `nn.BCEWithLogitsLoss` combines both operations into a single analytical formulation using the log-sum-exp identity:
    L = max(z, 0) - z*y + log(1 + exp(-|z|))
    This formulation is strictly guaranteed never to compute log(0) and is immune to numerical float32 overflow and underflow.

Q2: Why does Ultralytics YOLO11 use SGD with Momentum for large-scale training, but AdamW for fine-tuning on custom small datasets?
A2: - When training on massive datasets from scratch (COCO with 118,000 images):
      SGD with Momentum (0.937) exhibits a flatter curvature loss landscape bias that generalizes better out-of-distribution
      than adaptive methods, while requiring 50% less optimizer memory (storing only 1 velocity buffer vs 2 buffers for Adam).
    - When fine-tuning on custom, small, or domain-specific aerial drone datasets (e.g. 500 images of wind turbines):
      Adaptive learning rates per-parameter (AdamW) converge significantly faster, handle disparate gradient scales between
      frozen backbone and fresh detection heads, and decoupled weight decay prevents rapid overfitting on small sample sizes.
""")

if __name__ == "__main__":
    main()
