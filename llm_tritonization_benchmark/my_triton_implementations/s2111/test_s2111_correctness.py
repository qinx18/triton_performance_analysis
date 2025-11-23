#!/usr/bin/env python3
"""
Correctness Test for s2111
Tests: PyTorch baseline vs Triton LLM implementation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s2111_baseline import s2111_pytorch
    from llm_triton.s2111_triton_diagonal import s2111_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_correctness():
    """Test correctness across multiple sizes"""
    test_sizes = [100, 500, 1000]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s2111")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            # Initialize arrays
            aa = torch.randn(N, N, device='cuda', dtype=torch.float32)

            # Run PyTorch baseline
            pytorch_result = s2111_pytorch(aa.clone())

            # Run Triton LLM
            triton_result = s2111_triton(aa.clone())

            # Compare results (handle Inf values properly)
            if isinstance(pytorch_result, tuple):
                # Multiple outputs
                max_error = max([torch.max(torch.abs(p - t)).item()
                               for p, t in zip(pytorch_result, triton_result)])
            else:
                # Single output - compare only finite values to avoid Inf - Inf = NaN
                finite_mask = torch.isfinite(pytorch_result) & torch.isfinite(triton_result)
                diff = torch.abs(pytorch_result - triton_result)
                if finite_mask.any():
                    max_error = diff[finite_mask].max().item()
                else:
                    max_error = 0.0
                # Also check that Inf locations match
                if not torch.equal(torch.isinf(pytorch_result), torch.isinf(triton_result)):
                    max_error = float('inf')

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
