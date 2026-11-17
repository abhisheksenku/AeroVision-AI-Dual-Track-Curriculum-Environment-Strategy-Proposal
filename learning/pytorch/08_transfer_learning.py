"""
=============================================================================
AeroVision AI - Phase 2: Deep Learning with PyTorch
Module 08: Transfer Learning, Layer Freezing & Domain Adaptation
=============================================================================

Domain Problem:
Custom Drone Infrastructure Inspection Classifier:
Classifying high-value energy infrastructure from aerial drone camera feeds:
- Class 0: 'WIND_TURBINE_BLADE'
- Class 1: 'SOLAR_ARRAY'
- Class 2: 'HIGH_VOLTAGE_PYLON'

Annotating 100,000 drone images from scratch is economically unfeasible.
Transfer Learning allows us to take a deep backbone (e.g. MobileNetV2 with 3.5M parameters),
freeze its generic visual filters (edges, textures, geometric shapes), and train only a
lightweight custom classification head on a small dataset of specialized drone images.

Learning Objectives:
1. Understand the Transfer Learning spectrum: Feature Extraction vs Full Fine-Tuning.
2. Freezing Backbone Parameters: Setting `param.requires_grad = False`.
3. Replacing the Model Head: Custom `nn.Sequential` replacing `model.classifier`.
4. Parameter Budgeting: Demonstrating a 99% reduction in trainable parameters (3.5M -> 3.8K).
5. Differential Learning Rates: Passing different learning rates to backbone vs head.
6. Catastrophic Forgetting: Why fine-tuning with a high learning rate destroys pretrained knowledge.
7. Connection to AeroVision AI: How we fine-tune YOLO11 on VisDrone aerial datasets.
"""

import torch
import torch.nn as nn
import torchvision.models as models

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

def main():
    section_header("1. LOADING BACKBONE & INSPECTING BASE ARCHITECTURE")
    # Load MobileNetV2 architecture
    base_model = models.mobilenet_v2(weights=None)
    
    total_base_params = sum(p.numel() for p in base_model.parameters())
    print(f"Base MobileNetV2 Total Parameters: {total_base_params:,}")
    print(f"Original Classification Head:       {base_model.classifier}")
    print(f"Original Output Classes:            {base_model.classifier[1].out_features} (1,000 ImageNet categories)")

    # -------------------------------------------------------------------------
    section_header("2. FREEZING BACKBONE PARAMETERS (FEATURE EXTRACTION MODE)")
    # Freeze all 3.5M parameters in the feature extractor
    for param in base_model.features.parameters():
        param.requires_grad = False
        
    frozen_params = sum(p.numel() for p in base_model.features.parameters() if not p.requires_grad)
    print(f"Frozen Backbone Parameters: {frozen_params:,} (requires_grad = False)")

    # -------------------------------------------------------------------------
    section_header("3. ATTACHING CUSTOM DRONE INFRASTRUCTURE HEAD")
    # Input feature dimension from MobileNetV2 feature extractor is 1280
    in_features = base_model.classifier[1].in_features
    num_drone_classes = 3  # Turbine, Solar, Pylon
    
    # Custom head with Dropout + LayerNorm + Linear classifier
    custom_head = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 64),
        nn.SiLU(),
        nn.Linear(64, num_drone_classes)
    )
    
    # Replace default 1000-class head
    base_model.classifier = custom_head
    print(f"New Custom Drone Head Installed:\n{base_model.classifier}")

    # -------------------------------------------------------------------------
    section_header("4. PARAMETER BUDGET AUDIT")
    total_new_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    non_trainable_params = total_new_params - trainable_params
    
    print(f"New Model Total Parameters:       {total_new_params:,}")
    print(f"Non-Trainable (Frozen) Parameters:{non_trainable_params:,}")
    print(f"Trainable (Head Only) Parameters: {trainable_params:,}")
    print(f"Trainable Percentage of Network:  {(trainable_params / total_new_params) * 100:.2f}%")
    print(f"--> Slashed active gradient computation by {100 - (trainable_params / total_new_params) * 100:.2f}%!")

    # -------------------------------------------------------------------------
    section_header("5. DIFFERENTIAL LEARNING RATES FOR FINE-TUNING")
    # In Phase 2 of Transfer Learning (Fine-Tuning), we unfreeze the top backbone stages
    # but use a much smaller learning rate (e.g. 1e-5) on the backbone to avoid destroying
    # pretrained weights, while using a higher learning rate (e.g. 1e-3) on the new head.
    
    # Unfreeze only the last inverted residual block (stage 18)
    for param in base_model.features[-1].parameters():
        param.requires_grad = True
        
    fine_tune_params = [
        {"params": base_model.features[-1].parameters(), "lr": 1e-5},      # Gentle fine-tuning
        {"params": base_model.classifier.parameters(),    "lr": 1e-3}       # Active head learning
    ]
    
    optimizer = torch.optim.AdamW(fine_tune_params, weight_decay=0.01)
    print(f"Configured 2-tier Differential Learning Rate Optimizer:")
    print(f"  Stage 18 Feature Backbone LR: {optimizer.param_groups[0]['lr']} (Prevents Catastrophic Forgetting)")
    print(f"  Custom Drone Head LR:         {optimizer.param_groups[1]['lr']} (Rapid Task Adaptation)")

    # -------------------------------------------------------------------------
    section_header("6. VERIFICATION: FORWARD PASS ON AERIAL DRONE BATCH")
    # Simulated batch of 2 high-res aerial drone crops (Batch=2, Channels=3, H=224, W=224)
    drone_batch = torch.randn(2, 3, 224, 224)
    logits = base_model(drone_batch)
    probs = torch.softmax(logits, dim=-1)
    
    print(f"\nSimulated Forward Pass on Drone Batch {drone_batch.shape}:")
    print(f"  Output Logits Shape: {logits.shape}")
    print(f"  Crop 0 Probabilities: [Turbine: {probs[0,0]:.3f}, Solar: {probs[0,1]:.3f}, Pylon: {probs[0,2]:.3f}]")

    # -------------------------------------------------------------------------
    section_header("7. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: What is "Catastrophic Forgetting" in Transfer Learning, and what two engineering techniques prevent it?
A1: Catastrophic Forgetting occurs when a neural network fine-tuned on a new specialized dataset overwrites
    and destroys the rich, generalized visual representations learned during pre-training on millions of images.
    Prevention Techniques:
    1. Two-Stage Training: In Stage 1, completely freeze the backbone (`requires_grad = False`) and train ONLY
       the freshly initialized head until its random weights converge to reasonable gradients.
       Only in Stage 2 do you unfreeze the backbone. If you unfreeze too early, the massive random gradients
       from the untrained head propagate back into the backbone and scramble its pretrained feature filters.
    2. Discriminative / Differential Learning Rates: Assign a 10x to 100x smaller learning rate (e.g. 1e-5)
       to the lower backbone layers and a standard learning rate (e.g. 1e-3) to the classification head.

Q2: When should you use "Feature Extraction" versus "Full Fine-Tuning" for a drone analytics project?
A2: This decision depends on two factors: Dataset Size and Domain Similarity:
    1. Small Dataset + High Visual Similarity: Use Feature Extraction (Freeze backbone).
       (e.g., Detecting cars/trucks from low-altitude drones: COCO/ImageNet already has cars; training just the head
       avoids overfitting on small sample sizes).
    2. Large Dataset + Low Visual Similarity: Use Full Fine-Tuning with low LR.
       (e.g., Thermal/infrared drone inspection of solar panel micro-cracks: thermal signatures differ radically
       from standard RGB ImageNet photos; low-level backbone filters must adapt to thermal infrared radiometry).
""")

if __name__ == "__main__":
    main()
