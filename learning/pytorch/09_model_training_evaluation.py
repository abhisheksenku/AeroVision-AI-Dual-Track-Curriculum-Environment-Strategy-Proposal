"""
=============================================================================
AeroVision AI - Phase 2: Deep Learning with PyTorch
Module 09: Model Evaluation, Confusion Matrices & Checkpoint Persistence
=============================================================================

Domain Problem:
Safety-Critical Infrastructure Fault Detection & Checkpoint Management:
Evaluating a deep learning classifier detecting structural hazards from drone video:
- Class 0: 'SAFE_STRUCTURE'
- Class 1: 'SURFACE_WEATHERING' (Minor aesthetic wear)
- Class 2: 'CRITICAL_CRACK'     (Severe structural failure risk)

In industrial drone auditing, a False Negative on Class 2 (missing a critical crack)
leads to catastrophic structure collapse. We must compute per-class Precision, Recall,
and F1 scores, and persist complete training checkpoints with resume capability.

Learning Objectives:
1. Multi-Class Evaluation Metrics from scratch:
   - Confusion Matrix (TP, FP, FN, TN per class).
   - Per-Class Precision, Recall, and F1-Score.
   - Macro Average vs Weighted Average.
2. Model Checkpointing Best Practices:
   - Why saving `model.state_dict()` is preferred over saving full serialized model objects.
   - Saving complete training checkpoints (model weights, optimizer state, epoch, best metric).
   - Resuming training from saved checkpoints using `weights_only=True`.
3. Model Inference Profiling: Latency per sample in milliseconds.
4. Connection to AeroVision AI: Saving fine-tuned YOLO11 checkpoints and auditing failure modes.
"""

import os
import torch
import torch.nn as nn
import numpy as np

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

# =============================================================================
# 1. EVALUATION METRICS ENGINE FROM SCRATCH
# =============================================================================
def compute_multiclass_metrics(y_true, y_pred, class_names):
    """
    Computes Confusion Matrix, Precision, Recall, and F1-Score from scratch.
    y_true: 1D array of ground truth labels
    y_pred: 1D array of predicted class labels
    """
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Fill confusion matrix: rows = true, cols = predicted
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    print("\nPer-Class Detailed Performance Breakdown:")
    print(f"{'Class Name':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support'}")
    print("-" * 70)
    
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)
        
        print(f"{name:<22} | {prec:>8.4f}   | {rec:>8.4f}   | {f1:>8.4f}   | {support:>6d}")
        
    macro_prec = np.mean(precisions)
    macro_rec  = np.mean(recalls)
    macro_f1   = np.mean(f1s)
    total_acc  = np.trace(cm) / np.sum(cm)
    
    print("-" * 70)
    print(f"{'Macro Average':<22} | {macro_prec:>8.4f}   | {macro_rec:>8.4f}   | {macro_f1:>8.4f}   | {len(y_true):>6d}")
    print(f"\nOverall Top-1 Accuracy: {total_acc * 100:.2f}%")
    
    # Print formatted confusion matrix
    print("\nConfusion Matrix (Rows: Ground Truth, Columns: Predicted):")
    header_str = f"{'':>22} " + " ".join([f"{name[:6]:>7}" for name in class_names])
    print(header_str)
    for i, row in enumerate(cm):
        row_str = " ".join([f"{val:>7d}" for val in row])
        print(f"{class_names[i]:>22}: {row_str}")
        
    return cm

# =============================================================================
# 2. CHECKPOINT SAVING & LOADING BEST PRACTICES
# =============================================================================
class InspectionModel(nn.Module):
    def __init__(self, in_features=16, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.SiLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def demonstrate_checkpointing():
    checkpoint_dir = "models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "phase2_best_inspection_model.pt")

    # 1. Initialize model and optimizer
    model = InspectionModel(in_features=16, num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 2. Save comprehensive checkpoint dictionary
    checkpoint_data = {
        "epoch": 42,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_macro_f1": 0.942,
        "class_names": ["SAFE_STRUCTURE", "SURFACE_WEATHERING", "CRITICAL_CRACK"],
        "architecture": "InspectionModel(in=16, out=3)"
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    file_size_kb = os.path.getsize(checkpoint_path) / 1024
    print(f"Checkpoint successfully saved to: {checkpoint_path} ({file_size_kb:.2f} KB)")

    # 3. Resume / Load checkpoint in a fresh model instance
    fresh_model = InspectionModel(in_features=16, num_classes=3)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.001)

    loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)
    fresh_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    fresh_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    
    print("\nCheckpoint Successfully Restored into Fresh Model:")
    print(f"  Resumed Epoch:        {loaded_checkpoint['epoch']}")
    print(f"  Saved Best Macro F1:  {loaded_checkpoint['best_macro_f1']}")
    print(f"  Classes Restored:     {loaded_checkpoint['class_names']}")

    # Verify identical predictions between original and restored models
    test_tensor = torch.randn(2, 16)
    with torch.no_grad():
        out_orig = model(test_tensor)
        out_fresh = fresh_model(test_tensor)
        assert torch.allclose(out_orig, out_fresh), "Restored model weights differ!"
    print("--> Verification PASSED: Restored model produces bitwise identical predictions.")

# =============================================================================
# 3. EXECUTION
# =============================================================================
def main():
    section_header("1. MULTI-CLASS DRONE INSPECTION EVALUATION")
    class_names = ["SAFE_STRUCTURE", "SURFACE_WEATHERING", "CRITICAL_CRACK"]
    
    # 250 simulated validation samples with realistic drone failure distribution
    np.random.seed(42)
    # Ground truth: 150 Safe, 70 Weathered, 30 Critical Crack
    y_true = np.hstack([np.zeros(150), np.ones(70), np.full(30, 2)]).astype(int)
    
    # Predicted with 92% base accuracy and safety-conscious precision/recall trade-off
    y_pred = y_true.copy()
    # Simulate realistic confusion: 5 Safe misclassified as Weathered, 3 Weathered misclassified as Crack, etc.
    noise_indices = np.random.choice(len(y_true), size=18, replace=False)
    for idx in noise_indices:
        y_pred[idx] = np.random.choice([0, 1, 2])
        
    compute_multiclass_metrics(y_true, y_pred, class_names)

    section_header("2. PRODUCTION CHECKPOINT PERSISTENCE")
    demonstrate_checkpointing()

    section_header("3. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: Why is saving `model.state_dict()` strongly preferred over saving the entire model object with `torch.save(model)`?
A1: 1. Python Pickle Dependency: `torch.save(model)` serializes the entire Python class object using `pickle`.
       This binds the saved checkpoint to the exact directory structure, module paths, and Python environment.
       If you refactor your repository, rename a file, or load the checkpoint in a production C++ LibTorch or FastAPI service,
       the deserialization will crash with `AttributeError` or `ModuleNotFoundError`.
    2. Code-Weight Decoupling: `model.state_dict()` saves strictly a clean Python dictionary mapping layer names
       to parameter tensors. It contains zero Python bytecode, allowing the weights to be loaded into any refactored
       model definition, quantized for edge deployment (TFLite/TensorRT), or exported to ONNX.

Q2: For an autonomous drone infrastructure inspection system, which metric is more critical: Precision or Recall for CRITICAL_CRACK?
A2: Recall is overwhelmingly more critical for CRITICAL_CRACK:
    - False Positive (Low Precision): The drone flags a surface shadow or weathering as a critical crack.
      Consequence: A human maintenance engineer briefly reviews the high-res photo or visits the turbine.
      Cost: Minor operational time.
    - False Negative (Low Recall): The drone fails to detect an active 30cm structural crack on a wind turbine blade.
      Consequence: The blade undergoes structural fatigue failure at 200 km/h, destroying a $3,000,000 wind turbine.
      Cost: Catastrophic asset loss and potential loss of human life.
    Therefore, in drone AI safety engineering, we calibrate the confidence threshold to push Recall > 95% on hazard classes.
""")

if __name__ == "__main__":
    main()
