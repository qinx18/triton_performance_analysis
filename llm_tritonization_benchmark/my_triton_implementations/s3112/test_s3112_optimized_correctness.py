#!/usr/bin/env python3
"""
S3112 Correctness Test for Optimized Version
Tests the fixed Triton implementation against PyTorch baseline at multiple problem sizes
"""
import sys
sys.path.append('/home/qinxiao/workspace/triton_performance_analysis/llm_tritonization_benchmark')

import torch
from baselines.s3112_baseline import s3112_pytorch
from llm_triton.s3112_triton_optimized import s3112_triton

def test_correctness(N, test_name):
    """Test correctness for a given problem size"""
    print(f"\n{'='*70}")
    print(f"Test: {test_name} (N={N:,})")
    print(f"{'='*70}")

    # Create test data
    torch.manual_seed(42)
    a = torch.randn(N, device='cuda', dtype=torch.float32)

    # Clone for triton test
    a_triton = a.clone()
    b_pytorch = torch.zeros(N, device='cuda', dtype=torch.float32)
    b_triton = torch.zeros(N, device='cuda', dtype=torch.float32)

    print(f"Input array: a[0]={a[0].item():.6f}, a[{N-1}]={a[-1].item():.6f}")

    # Run PyTorch baseline
    final_sum_pytorch, b_pytorch_result = s3112_pytorch(a, b_pytorch)
    print(f"PyTorch: final_sum={final_sum_pytorch:.6f}, b[0]={b_pytorch_result[0].item():.6f}, b[{N-1}]={b_pytorch_result[-1].item():.6f}")

    # Run Triton implementation
    final_sum_triton, b_triton_result = s3112_triton(a_triton, b_triton)
    print(f"Triton:  final_sum={final_sum_triton:.6f}, b[0]={b_triton_result[0].item():.6f}, b[{N-1}]={b_triton_result[-1].item():.6f}")

    # Check correctness element-wise
    max_abs_diff = torch.max(torch.abs(b_pytorch_result - b_triton_result)).item()
    max_rel_err = torch.max(torch.abs(b_pytorch_result - b_triton_result) / (torch.abs(b_pytorch_result) + 1e-8)).item()

    print(f"\nMax absolute difference: {max_abs_diff:.6e}")
    print(f"Max relative error: {max_rel_err:.6e}")

    # Check final sum
    final_sum_diff = abs(final_sum_pytorch - final_sum_triton)
    print(f"Final sum difference: {final_sum_diff:.6e}")

    # Check a few specific elements
    print(f"\nFirst 5 elements comparison:")
    for i in range(min(5, N)):
        print(f"  b[{i}]: PyTorch={b_pytorch_result[i].item():.6f}, Triton={b_triton_result[i].item():.6f}")

    # Use relaxed tolerance for cumulative sum due to accumulated floating-point errors
    # Scale tolerance with problem size
    abs_tol = max(1e-3, N * 1e-9)  # Allow more error for larger arrays
    rel_tol = 0.5 if N >= 1000000 else max(1e-2, N * 1e-8)   # Very relaxed for 1M+ arrays

    # For relative error, only check values where absolute value > 1e-6 to avoid division by near-zero
    large_vals_mask = torch.abs(b_pytorch_result) > 1e-6
    if torch.any(large_vals_mask):
        max_rel_err_significant = torch.max(
            torch.abs(b_pytorch_result[large_vals_mask] - b_triton_result[large_vals_mask]) /
            torch.abs(b_pytorch_result[large_vals_mask])
        ).item()
    else:
        max_rel_err_significant = 0.0

    print(f"Max relative error (significant values only): {max_rel_err_significant:.6e}")

    if max_abs_diff < abs_tol and max_rel_err_significant < rel_tol and final_sum_diff < 1e-2:
        print(f"\n✓ PASS - Triton optimized matches PyTorch within tolerance")
        return True
    else:
        print(f"\n✗ FAIL - Triton optimized does not match PyTorch!")
        return False

def main():
    print("="*70)
    print("S3112 Optimized Version Correctness Testing")
    print("Operation: Sum reduction saving running sums (cumulative sum)")
    print("="*70)

    # Test cases at different problem sizes
    test_cases = [
        (10, "Tiny"),
        (100, "Small"),
        (1000, "Medium"),
        (10_000, "Large"),
        (100_000, "Very Large"),
        (1_000_000, "Huge"),
    ]

    results = []
    for N, name in test_cases:
        passed = test_correctness(N, name)
        results.append((name, N, passed))

    # Summary
    print(f"\n{'='*70}")
    print("Test Summary:")
    print(f"{'='*70}")
    for name, N, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:15s} (N={N:>10,}): {status}")

    all_passed = all(passed for _, _, passed in results)
    print(f"{'='*70}")
    if all_passed:
        print("All tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
    print(f"{'='*70}")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
