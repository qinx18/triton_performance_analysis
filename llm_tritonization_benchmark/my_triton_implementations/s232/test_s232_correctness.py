#!/usr/bin/env python3
"""
Correctness Test for s232
Tests: PyTorch baseline vs Triton LLM implementation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s232_baseline import s232_pytorch
    from llm_triton.s232_triton_llm import s232_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_correctness():
    """Test correctness across multiple sizes"""
    # Reduced sizes to avoid timeout (baseline uses slow Python nested loops for 2D triangular pattern)
    # Triangular pattern: j from 1 to N, i from 1 to j → O(N²) with Python loops
    test_sizes = [100, 200, 500]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s232")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            # Initialize arrays (2D arrays for triangular dependency pattern)
            aa = torch.randn(N, N, device='cuda', dtype=torch.float32)
            bb = torch.randn(N, N, device='cuda', dtype=torch.float32)

            # Run PyTorch baseline
            pytorch_result = s232_pytorch(aa.clone(), bb.clone())

            # Run Triton LLM
            triton_result = s232_triton(aa.clone(), bb.clone())

            # Compare results (s232 squares values → exponential growth → need relative tolerance)
            if isinstance(pytorch_result, tuple):
                pytorch_result = pytorch_result[0] if len(pytorch_result) == 1 else pytorch_result
                triton_result = triton_result[0] if len(triton_result) == 1 else triton_result

            # Only compare finite values (algorithm overflows to Inf)
            finite_mask = torch.isfinite(pytorch_result) & torch.isfinite(triton_result)
            pytorch_finite = pytorch_result[finite_mask]
            triton_finite = triton_result[finite_mask]

            # Check Inf locations match
            inf_match = torch.equal(torch.isinf(pytorch_result), torch.isinf(triton_result))

            if pytorch_finite.numel() > 0:
                # Use relative tolerance for exponentially growing values
                rel_error = torch.max(torch.abs((pytorch_finite - triton_finite) /
                                               (torch.abs(pytorch_finite) + 1e-10))).item()

                if rel_error < 1e-3 and inf_match:
                    print(f"✓ PASS  (rel_err={rel_error:.2e}, inf_match={inf_match})")
                else:
                    print(f"✗ FAIL  (rel_err={rel_error:.2e}, inf_match={inf_match})")
                    all_passed = False
            else:
                print(f"✗ FAIL  (all values are Inf/NaN)")
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
