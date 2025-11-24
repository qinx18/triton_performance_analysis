#!/usr/bin/env python3
"""
Benchmark s424: Multi-kernel vs Single-kernel versions
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import time

from baselines.s424_baseline_correct import s424_pytorch
from llm_triton.s424_triton_correct import s424_triton as s424_multikernel
from llm_triton.s424_triton_optimized import s424_triton as s424_onekernel

def benchmark(func, a, flat_2d_array, name, warmup=10, iters=50):
    """Benchmark a function"""
    # Warmup
    for _ in range(warmup):
        result = func(a.clone(), flat_2d_array.clone())
    torch.cuda.synchronize()

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        result = func(a.clone(), flat_2d_array.clone())
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
    avg_time = elapsed_time / iters
    return avg_time

print("="*80)
print("s424 Performance Benchmark: Multi-kernel vs Single-kernel")
print("="*80)
print()

sizes = [100, 500, 1000, 5000, 10000]

print(f"{'N':<8} {'Elements':<12} {'Strips':<8} {'PyTorch':<12} {'Multi-K':<12} {'Single-K':<12} {'Speedup':<10}")
print("-"*80)

for N in sizes:
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    flat_2d_array = torch.randn(N * N, device='cuda', dtype=torch.float32)

    # Calculate metadata
    n_elements = N - 1
    num_strips = (n_elements + 63) // 64

    # Benchmark
    pytorch_time = benchmark(s424_pytorch, a, flat_2d_array, "PyTorch")
    multi_time = benchmark(s424_multikernel, a, flat_2d_array, "Multi-kernel")
    single_time = benchmark(s424_onekernel, a, flat_2d_array, "Single-kernel")

    speedup = multi_time / single_time

    print(f"{N:<8} {n_elements:<12} {num_strips:<8} {pytorch_time:<12.4f} {multi_time:<12.4f} {single_time:<12.4f} {speedup:<10.2f}x")

print("="*80)
print("\nAnalysis:")
print("-"*80)
print("Multi-kernel: Launches one kernel per strip from Python loop")
print("Single-kernel: Launches ONE kernel, loop over strips is inside kernel")
print()
print("Expected: Single-kernel should be faster due to reduced launch overhead")
print("          Benefit increases with more strips (larger arrays)")
