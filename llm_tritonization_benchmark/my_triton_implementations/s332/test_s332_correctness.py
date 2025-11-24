#!/usr/bin/env python3
"""
Correctness Test for s332
Tests: PyTorch baseline vs Triton LLM implementation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s332_baseline import s332_pytorch
    from llm_triton.s332_triton_correct import s332_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_correctness():
    """Test correctness across multiple sizes"""
    test_sizes = [100, 1000, 10000]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s332")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            # Initialize arrays
            a = torch.randn(N, device='cuda', dtype=torch.float32)
            t = 0.5  # Scalar parameter for threshold

            # Run PyTorch baseline
            pytorch_result = s332_pytorch(a.clone(), t)

            # Run Triton LLM
            triton_result = s332_triton(a.clone(), t)

            # Compare results
            if isinstance(pytorch_result, tuple):
                # Multiple outputs
                max_error = max([torch.max(torch.abs(p - t)).item()
                               for p, t in zip(pytorch_result, triton_result)])
            else:
                # Single output
                max_error = torch.max(torch.abs(pytorch_result - triton_result)).item()

            # Check if within tolerance
            if max_error < 1e-3:  # Relaxed tolerance for complex functions
                print(f"✓ PASS  (max_err={max_error:.2e})")
            else:
                print(f"✗ FAIL  (max_error={max_error:.2e})")
                all_passed = False

        except Exception as e:
            print(f"✗ ERROR: {e}")
            all_passed = False

    print("="*70)
    if all_passed:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED!")
    print("="*70)

    return all_passed

if __name__ == "__main__":
    success = test_correctness()
    sys.exit(0 if success else 1)
