#!/usr/bin/env python3
"""
Verify correctness of s2102_triton_v2 and compare performance with v1
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import time

from baselines.s2102_baseline import s2102_pytorch
from my_triton_implementations.s2102.s2102_triton import s2102_triton as s2102_v1
from my_triton_implementations.s2102.s2102_triton_v2 import s2102_triton as s2102_v2

def test_correctness():
    """Test correctness of v2 implementation"""
    test_sizes = [100, 1000, 10000]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s2102_triton_v2")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            # Initialize arrays
            aa = torch.randn(N, N, device='cuda', dtype=torch.float32)

            # Run PyTorch baseline
            pytorch_result = s2102_pytorch(aa.clone())

            # Run Triton v2
            triton_result = s2102_v2(aa.clone())

            # Compare results
            max_error = torch.max(torch.abs(pytorch_result - triton_result)).item()

            # Check if within tolerance
            if max_error < 1e-3:
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
    print()

    return all_passed

def benchmark_performance():
    """Compare performance between v1 and v2"""
    test_sizes = [100, 500, 1000, 5000, 10000]
    num_warmup = 10
    num_iterations = 100

    print("="*70)
    print(f"Performance Comparison: s2102_v1 vs s2102_v2")
    print("="*70)
    print(f"{'Size':>6} | {'v1 (ms)':>10} | {'v2 (ms)':>10} | {'Speedup':>10}")
    print("-"*70)

    for N in test_sizes:
        aa = torch.randn(N, N, device='cuda', dtype=torch.float32)

        # Warmup for v1
        for _ in range(num_warmup):
            s2102_v1(aa.clone())
        torch.cuda.synchronize()

        # Benchmark v1
        start = time.perf_counter()
        for _ in range(num_iterations):
            s2102_v1(aa.clone())
        torch.cuda.synchronize()
        time_v1 = (time.perf_counter() - start) / num_iterations * 1000  # ms

        # Warmup for v2
        for _ in range(num_warmup):
            s2102_v2(aa.clone())
        torch.cuda.synchronize()

        # Benchmark v2
        start = time.perf_counter()
        for _ in range(num_iterations):
            s2102_v2(aa.clone())
        torch.cuda.synchronize()
        time_v2 = (time.perf_counter() - start) / num_iterations * 1000  # ms

        speedup = time_v1 / time_v2
        speedup_str = f"{speedup:.2f}x" if speedup >= 1 else f"{1/speedup:.2f}x slower"

        print(f"{N:>6} | {time_v1:>10.4f} | {time_v2:>10.4f} | {speedup_str:>10}")

    print("="*70)

if __name__ == "__main__":
    # First verify correctness
    if test_correctness():
        # Then benchmark performance
        benchmark_performance()
    else:
        print("Skipping performance comparison due to correctness failures")
        sys.exit(1)
