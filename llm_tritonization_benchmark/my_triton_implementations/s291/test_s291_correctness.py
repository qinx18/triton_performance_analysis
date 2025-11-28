#!/usr/bin/env python3
"""
Correctness Test for s291
Tests: PyTorch baseline vs Triton LLM implementation (in-place comparison)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s291_baseline import s291_pytorch
    from llm_triton.s291_triton_llm_v3 import s291_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_correctness():
    """Test correctness across multiple sizes"""
    test_sizes = [100, 1000, 10000]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s291")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            # Initialize base arrays
            a = torch.randn(N, device='cuda', dtype=torch.float32)
            b = torch.randn(N, device='cuda', dtype=torch.float32)
            iterations = 1  # Scalar parameter (integer)

            # Create copies for PyTorch baseline
            a_pt = a.clone()
            b_pt = b.clone()

            # Create copies for Triton implementation
            a_tr = a.clone()
            b_tr = b.clone()

            # Run PyTorch baseline (may modify arrays in-place or return result)
            pytorch_result = s291_pytorch(a_pt, b_pt, iterations)

            # Run Triton LLM (modifies arrays in-place)
            s291_triton(a_tr, b_tr, iterations)

            # Compare output arrays directly (in-place modification)
            max_error = torch.max(torch.abs(a_pt - a_tr)).item()

            # Check if within tolerance
            if max_error < 1e-3:  # Relaxed tolerance for complex functions
                print(f"✓ PASS  (max_err={max_error:.2e})")
            else:
                print(f"✗ FAIL  (max_error={max_error:.2e})")
                all_passed = False

        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
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
