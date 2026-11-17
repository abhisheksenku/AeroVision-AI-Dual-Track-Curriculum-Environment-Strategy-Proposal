"""
=============================================================================
AeroVision AI - Phase 2: Deep Learning with PyTorch
Module 02: Autograd Engine, Dynamic Graphs & Gradient Accumulation
=============================================================================

Domain Problem:
Custom Loss Backpropagation for Aerial Bounding Box Regression:
When optimizing bounding box coordinates (center_x, center_y, width, height) under
IoU or CIoU loss, PyTorch dynamically builds an acyclic graph of operations.
Understanding Autograd, gradient accumulation, graph retention, and inference mode
is essential to prevent GPU/CPU memory leaks and debug vanishing/exploding gradients.

Learning Objectives:
1. Understand PyTorch's Dynamic Computational Graph (Define-by-Run).
2. Leaf nodes vs Non-leaf nodes and the grad_fn backward execution chain.
3. The Chain Rule in Autograd: Computing d(Loss)/dx via backward().
4. Gradient Accumulation: Why .grad accumulates by default and why zero_grad() is mandatory.
5. Context Managers: torch.no_grad() and torch.inference_mode() for production inference.
6. Graph Detachment: tensor.detach() for stopping gradient flow (e.g. Teacher-Student models).
7. Gradient Verification: Finite Difference Numerical Approximation vs Analytical Autograd.
8. Connection to AeroVision AI: Preventing memory leaks during real-time 30 FPS video streaming.
"""

import torch

def section_header(title):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)

def main():
    section_header("1. DYNAMIC COMPUTATIONAL GRAPH & LEAF NODES")
    # Leaf node: Tensor created directly by the user (requires_grad=True)
    # Non-leaf node: Tensor created as the result of a mathematical operation
    x = torch.tensor([2.0, 4.0], requires_grad=True)  # Leaf
    w = torch.tensor([3.0, 5.0], requires_grad=True)  # Leaf
    b = torch.tensor(1.5, requires_grad=True)          # Leaf

    # Forward computation
    y = x * w + b                    # Non-leaf: grad_fn = <AddBackward0>
    loss = torch.sum(y ** 2)         # Non-leaf: grad_fn = <SumBackward0>

    print(f"Leaf tensor x:      requires_grad={x.requires_grad}, is_leaf={x.is_leaf}, grad_fn={x.grad_fn}")
    print(f"Intermediate y:     requires_grad={y.requires_grad}, is_leaf={y.is_leaf}, grad_fn={y.grad_fn}")
    print(f"Scalar Loss:        value={loss.item():.2f}, grad_fn={loss.grad_fn}")

    # -------------------------------------------------------------------------
    section_header("2. BACKWARD PASS & THE MULTIVARIATE CHAIN RULE")
    # Math:
    # y_i = x_i * w_i + b
    # loss = sum(y_i^2) = (x_0*w_0 + b)^2 + (x_1*w_1 + b)^2
    # y_0 = 2.0 * 3.0 + 1.5 = 7.5  --> y_0^2 = 56.25
    # y_1 = 4.0 * 5.0 + 1.5 = 21.5 --> y_1^2 = 462.25
    # loss = 56.25 + 462.25 = 518.50
    #
    # Analytical Gradients:
    # d(loss)/dx_i = 2 * y_i * w_i
    # d(loss)/dx_0 = 2 * 7.5 * 3.0 = 45.0
    # d(loss)/dx_1 = 2 * 21.5 * 5.0 = 215.0
    # d(loss)/db   = 2 * y_0 * 1 + 2 * y_1 * 1 = 2 * (7.5 + 21.5) = 58.0

    loss.backward()

    print(f"Loss value:          {loss.item():.2f} (Expected: 518.50)")
    print(f"Computed x.grad:     {x.grad.tolist()} (Expected: [45.0, 215.0])")
    print(f"Computed w.grad:     {w.grad.tolist()}")
    print(f"Computed b.grad:     {b.grad.item():.2f} (Expected: 58.00)")

    assert torch.allclose(x.grad, torch.tensor([45.0, 215.0])), "Gradient verification failed!"
    assert torch.isclose(b.grad, torch.tensor(58.0)), "Bias gradient verification failed!"
    print("--> Analytical Autograd verification PASSED!")

    # -------------------------------------------------------------------------
    section_header("3. GRADIENT ACCUMULATION & WHY ZERO_GRAD IS MANDATORY")
    # In PyTorch, calling .backward() ACCUMULATES (adds) gradients onto .grad:
    # param.grad += new_gradient
    # This design exists so engineers can implement gradient accumulation across
    # multiple mini-batches when GPU/CPU RAM cannot fit a large batch size.
    # However, if not manually cleared, gradients from previous batches corrupt training!

    print(f"x.grad before second backward: {x.grad.tolist()}")

    # Forward pass again
    y2 = x * w + b
    loss2 = torch.sum(y2 ** 2)
    loss2.backward()

    print(f"x.grad AFTER second backward without zeroing: {x.grad.tolist()} (Doubled!)")
    assert torch.allclose(x.grad, torch.tensor([90.0, 430.0])), "Accumulation verification failed!"

    # Correct practice: manually zero gradients
    x.grad.zero_()
    w.grad.zero_()
    b.grad.zero_()
    print(f"x.grad after manual zero_(): {x.grad.tolist()} (Cleaned)")

    # -------------------------------------------------------------------------
    section_header("4. INFERENCE CONTEXT: TORCH.NO_GRAD() & INFERENCE_MODE()")
    # During model deployment (AeroVision video streamer), tracking gradients wastes 50%+ RAM
    # and burns CPU cycles building graph structures that are never backpropagated.
    x_test = torch.randn(1000, 1000, requires_grad=True)

    with torch.no_grad():
        y_test = x_test * 2.5 + 1.0
        print(f"Under torch.no_grad(): y_test.requires_grad = {y_test.requires_grad}, grad_fn = {y_test.grad_fn}")

    # PyTorch 1.9+ introduces torch.inference_mode(): even faster than no_grad
    with torch.inference_mode():
        y_inf = x_test * 2.5 + 1.0
        print(f"Under torch.inference_mode(): y_inf.requires_grad = {y_inf.requires_grad}")

    # -------------------------------------------------------------------------
    section_header("5. DETACHING TENSORS: GRAPH PRUNING")
    # .detach() creates a new tensor that shares the storage but has requires_grad=False.
    # Used in teacher-student architectures, target networks, and object tracking.
    x_frozen = torch.tensor([5.0], requires_grad=True)
    x_active = torch.tensor([10.0], requires_grad=True)

    # Detach x_frozen branch: treated as a numerical constant
    feat_frozen = (x_frozen ** 2).detach()
    feat_active = x_active ** 2

    # Combined loss
    loss_combo = feat_frozen * 3.0 + feat_active
    loss_combo.backward()

    print(f"Gradient on x_frozen (detached): {x_frozen.grad} (None: gradient chain stopped at detach!)")
    print(f"Gradient on x_active (attached): {x_active.grad.item():.2f} (Expected: 2 * 10 = 20.0)")
    assert x_frozen.grad is None, "Frozen tensor should have zero gradient!"
    assert x_active.grad.item() == 20.0, "Active gradient should be 20.0!"

    # -------------------------------------------------------------------------
    section_header("6. FINITE DIFFERENCE GRADIENT CHECKING (NUMERICAL VS AUTOGRAD)")
    # Numerical derivative formula:
    # f'(x) approx [f(x + eps) - f(x - eps)] / (2 * eps)
    # Note: Finite difference checking requires float64 (double precision)
    # because float32 precision limits introduce rounding errors at eps = 1e-6.
    def test_func(val):
        return (val ** 3) + torch.sin(val)

    x_val = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    out = test_func(x_val)
    out.backward()
    analytical_grad = x_val.grad.item()

    # Numerical approximation with eps = 1e-6 in float64
    eps = 1e-6
    f_plus = test_func(torch.tensor([1.5 + eps], dtype=torch.float64)).item()
    f_minus = test_func(torch.tensor([1.5 - eps], dtype=torch.float64)).item()
    numerical_grad = (f_plus - f_minus) / (2 * eps)

    print(f"Analytical Autograd Gradient: {analytical_grad:.8f}")
    print(f"Numerical Finite Diff Grad:   {numerical_grad:.8f}")
    diff = abs(analytical_grad - numerical_grad)
    print(f"Absolute Gradient Discrepancy: {diff:.10f}")
    assert diff < 1e-6, "Gradient check failed!"
    print("--> Float64 Numerical Finite-Difference vs Autograd PASSED (< 1e-6 error)!")

    # -------------------------------------------------------------------------
    section_header("7. INTERVIEW QUESTIONS & MODEL ANSWERS")
    print("""
Q1: Why does PyTorch accumulate gradients by default instead of overwriting them on each backward() pass?
A1: Gradient accumulation is an intentional architectural feature in PyTorch designed for memory-constrained training:
    - If training a high-resolution drone vision model (e.g. 1280x1280) requires an effective batch size of 64,
      but system RAM/VRAM can only fit a batch size of 8 before encountering an Out-Of-Memory (OOM) error,
      an engineer can run 8 forward/backward passes sequentially without calling `optimizer.zero_grad()`.
    - PyTorch accumulates the gradients from each mini-batch (`param.grad += g_i`).
    - After 8 iterations, calling `optimizer.step()` and `optimizer.zero_grad()` updates parameters
      with the exact mathematical equivalent of an effective batch size of 64 without ever needing 64 frames in memory at once!

Q2: What is the physical difference between `tensor.detach()` and `torch.no_grad()`?
A2: - `torch.no_grad()` is a dynamic thread-local context manager that disables the Autograd engine entirely for all
      operations executed within the block. Tensors produced have requires_grad=False and zero graph memory is allocated.
    - `tensor.detach()` operates on an existing tensor with an existing graph history: it returns a new view of the tensor
      sharing the same storage, but severed from the computational graph (grad_fn=None), allowing its numerical values
      to be used in a downstream calculation without gradients flowing backward through its ancestral operations.
""")

if __name__ == "__main__":
    main()
