#!/usr/bin/env python3
"""
Correctness Test for s452
Tests: PyTorch baseline vs Triton LLM implementation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s452_baseline import s452_pytorch
    from llm_triton.s452_triton_correct import s452_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_correctness():
    """Test correctness across multiple sizes"""
    test_sizes = [100, 1000, 10000]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s452")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            # Initialize arrays
            a = torch.randn(N, device='cuda', dtype=torch.float32)
            b = torch.randn(N, device='cuda', dtype=torch.float32)
            c = torch.randn(N, device='cuda', dtype=torch.float32)

            # Run PyTorch baseline
            pytorch_result = s452_pytorch(a.clone(), b.clone(), c.clone())

            # Run Triton LLM
            triton_result = s452_triton(a.clone(), b.clone(), c.clone())

            # Compare results
            if isinstance(pytorch_result, tuple):
                # Multiple outputs
                max_error = max([torch.max(torch.abs(p - t)).item()
                               for p, t in zip(pytorch_result, triton_result)])
            else:
                # Single output
                max_error = torch.max(torch.abs(pytorch_result - triton_result)).item()

            # Check if within tolerance
            # Use 1e-2 for s452 because multiplying by large indices (up to 10000)
            # causes acceptable float32 precision loss
            if max_error < 1e-2:  # Tolerance adjusted for operations with large integers
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
