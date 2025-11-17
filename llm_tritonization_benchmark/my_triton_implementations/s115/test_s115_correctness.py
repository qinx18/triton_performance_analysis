#!/usr/bin/env python3
"""
Correctness Test for s115
Tests: PyTorch baseline vs Triton LLM implementation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s115_baseline import s115_pytorch
    from llm_triton.s115_triton_llm import s115_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_correctness():
    """Test correctness across multiple sizes"""
    # Reduced test sizes for s115 since it requires N sequential kernel launches
    test_sizes = [10, 50, 100]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s115")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            # Initialize arrays
            a = torch.randn(N, device='cuda', dtype=torch.float32)
            aa = torch.randn(N, N, device='cuda', dtype=torch.float32)

            # Run PyTorch baseline
            pytorch_result = s115_pytorch(a.clone(), aa.clone())

            # Run Triton LLM
            triton_result = s115_triton(a.clone(), aa.clone())

            # Compare results using relative tolerance
            # s115 (back substitution) can cause values to grow exponentially,
            # so we need relative tolerance instead of absolute
            # Using rtol=1e-4 (0.01%) for N sequential operations with accumulation
            if isinstance(pytorch_result, tuple):
                # Multiple outputs
                passed = all([torch.allclose(p, t, rtol=1e-4, atol=1e-6)
                             for p, t in zip(pytorch_result, triton_result)])
                # Calculate max relative error for reporting
                abs_diffs = [torch.abs(p - t) for p, t in zip(pytorch_result, triton_result)]
                max_mags = [torch.maximum(torch.abs(p), torch.abs(t))
                           for p, t in zip(pytorch_result, triton_result)]
                rel_errors = [torch.where(mag > 1e-10, diff / mag, diff)
                             for diff, mag in zip(abs_diffs, max_mags)]
                max_rel_error = max([torch.max(rel_err).item() for rel_err in rel_errors])
            else:
                # Single output
                passed = torch.allclose(pytorch_result, triton_result, rtol=1e-4, atol=1e-6)
                # Calculate max relative error for reporting
                abs_diff = torch.abs(pytorch_result - triton_result)
                max_magnitude = torch.maximum(torch.abs(pytorch_result), torch.abs(triton_result))
                rel_error = torch.where(max_magnitude > 1e-10, abs_diff / max_magnitude, abs_diff)
                max_rel_error = torch.max(rel_error).item()

            # Check if within tolerance
            if passed:
                print(f"✓ PASS  (max_rel_err={max_rel_error:.2e})")
            else:
                print(f"✗ FAIL  (max_rel_error={max_rel_error:.2e})")
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
