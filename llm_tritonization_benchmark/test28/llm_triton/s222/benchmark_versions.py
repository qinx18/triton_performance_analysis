#!/usr/bin/env python3
"""Benchmark different s222 implementations."""

import sys
sys.path.insert(0, "/home/qinxiao/workspace/triton_performance_analysis/llm_tritonization_benchmark")

import torch
import time
import numpy as np

from c_reference.tsvc_all_reference import s222_c
from test28.llm_triton.s222.attempt1 import s222_triton as s222_original
from test28.llm_triton.s222.s222_optimal import s222_triton_v1, s222_triton_v2, s222_triton_v3

def benchmark_func(func, a, b, c, e, name, warmup=10, runs=100, is_cpu=False):
    """Benchmark a function."""
    # Clone for each run
    for _ in range(warmup):
        if is_cpu:
            a_t, b_t, c_t, e_t = a.cpu().clone(), b.cpu().clone(), c.cpu().clone(), e.cpu().clone()
        else:
            a_t, b_t, c_t, e_t = a.clone(), b.clone(), c.clone(), e.clone()
        func(a_t, b_t, c_t, e_t)
    if not is_cpu:
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if is_cpu:
            a_t, b_t, c_t, e_t = a.cpu().clone(), b.cpu().clone(), c.cpu().clone(), e.cpu().clone()
        else:
            a_t, b_t, c_t, e_t = a.clone(), b.clone(), c.clone(), e.clone()
            torch.cuda.synchronize()
        start = time.perf_counter()
        func(a_t, b_t, c_t, e_t)
        if not is_cpu:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    return avg_ms, std_ms

def check_correctness(func, a, b, c, e, name):
    """Check if function produces correct results."""
    # C reference needs CPU tensors
    a_ref, b_ref, c_ref, e_ref = a.cpu().clone(), b.cpu().clone(), c.cpu().clone(), e.cpu().clone()
    # GPU version uses CUDA tensors
    a_test, b_test, c_test, e_test = a.clone(), b.clone(), c.clone(), e.clone()

    s222_c(a_ref, b_ref, c_ref, e_ref)
    func(a_test, b_test, c_test, e_test)

    # Move GPU results to CPU for comparison
    e_test_cpu = e_test.cpu()
    a_test_cpu = a_test.cpu()

    # Check e array (the only one that changes)
    e_match = torch.allclose(e_ref, e_test_cpu, rtol=1e-4, atol=1e-4)
    a_match = torch.allclose(a_ref, a_test_cpu, rtol=1e-4, atol=1e-4)

    if e_match and a_match:
        return True, None
    else:
        e_err = torch.max(torch.abs(e_ref - e_test_cpu)).item()
        a_err = torch.max(torch.abs(a_ref - a_test_cpu)).item()
        return False, f"e_err={e_err:.2e}, a_err={a_err:.2e}"

def main():
    N = 32000
    torch.manual_seed(42)

    # Initialize arrays
    a = torch.rand(N, dtype=torch.float32, device='cuda')
    b = torch.rand(N, dtype=torch.float32, device='cuda')
    c = torch.rand(N, dtype=torch.float32, device='cuda')
    # e needs special init to avoid overflow: start with small values
    e = torch.full((N,), 0.99, dtype=torch.float32, device='cuda')

    print("=" * 70)
    print("s222 Benchmark: Comparing different implementations")
    print("=" * 70)
    print(f"Array size: {N}")
    print()

    # Benchmark C reference (needs CPU tensors)
    print("Benchmarking C reference...")
    c_time, c_std = benchmark_func(s222_c, a, b, c, e, "C reference", is_cpu=True)
    print(f"  C reference:    {c_time:8.3f} ms (±{c_std:.3f})")
    print()

    versions = [
        ("Original LLM", s222_original),
        ("V1: PyTorch loop", s222_triton_v1),
        ("V2: Minimal Triton", s222_triton_v2),
        ("V3: Parallel pow", s222_triton_v3),
    ]

    print("Checking correctness and benchmarking...")
    print("-" * 70)
    print(f"{'Version':<25} {'Correct':<10} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    for name, func in versions:
        # Check correctness
        correct, err = check_correctness(func, a, b, c, e, name)

        if correct:
            # Benchmark
            t, std = benchmark_func(func, a, b, c, e, name)
            speedup = c_time / t
            print(f"{name:<25} {'PASS':<10} {t:>8.3f} ±{std:.3f}  {speedup:>8.2f}x")
        else:
            print(f"{name:<25} {'FAIL':<10} {err}")

    print("-" * 70)
    print(f"C reference time: {c_time:.3f} ms")

if __name__ == "__main__":
    main()
