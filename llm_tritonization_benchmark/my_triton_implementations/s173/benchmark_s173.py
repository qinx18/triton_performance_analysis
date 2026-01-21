#!/usr/bin/env python3
"""
Performance Benchmark for s173
Compares Triton implementation against original TSVC C reference.
"""
import sys
import time
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    from c_reference.tsvc_all_reference import s173_c
    from test23.llm_triton.s173.attempt1 import s173_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def get_func_params(func):
    sig = inspect.signature(func)
    return list(sig.parameters.keys())

def build_args(func, available_tensors, available_scalars):
    params = get_func_params(func)
    args = []
    for p in params:
        if p in available_tensors:
            args.append(available_tensors[p])
        elif p in available_scalars:
            args.append(available_scalars[p])
    return args

def benchmark():
    N = 32000
    num_warmup = 10
    num_iterations = 100
    timeout_per_section = 60  # 60 seconds per section (warmup/benchmark)

    print("="*70)
    print(f"Performance Benchmark: s173")
    print(f"Comparing Triton (GPU) vs TSVC C reference (CPU)")
    print(f"Array size: N={N}")
    print("="*70)

    # Initialize arrays on GPU
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    iterations = 1
    k = 0

    # Create numpy arrays for C reference (on CPU)
    c_arrays = {"a": a.cpu().numpy().copy(), "b": b.cpu().numpy().copy()}
    tr_tensors = {"a": a.clone(), "b": b.clone()}
    scalars = {"iterations": iterations, "k": k}

    c_args = build_args(s173_c, c_arrays, scalars)
    tr_args = build_args(s173_triton, tr_tensors, scalars)

    c_time = None
    tr_time = None

    # Benchmark C reference (CPU, with separate timeout handling)
    try:
        print(f"Warming up C reference ({num_warmup} iterations)...")
        start_time = time.perf_counter()
        for i in range(num_warmup):
            if time.perf_counter() - start_time > timeout_per_section:
                raise TimeoutError("C reference warmup timeout")
            # Reset arrays for each iteration
            for arr in c_arrays:
                c_arrays[arr] = c_arrays[arr].copy()
            c_args = build_args(s173_c, c_arrays, scalars)
            s173_c(*c_args)

        print(f"Benchmarking C reference ({num_iterations} iterations)...")
        c_start = time.perf_counter()
        bench_start = time.perf_counter()
        for i in range(num_iterations):
            if time.perf_counter() - bench_start > timeout_per_section:
                raise TimeoutError("C reference benchmark timeout")
            for arr in c_arrays:
                c_arrays[arr] = c_arrays[arr].copy()
            c_args = build_args(s173_c, c_arrays, scalars)
            s173_c(*c_args)
        c_time = (time.perf_counter() - c_start) / num_iterations
        print(f"  C reference time: {c_time*1000:.3f} ms")
    except (TimeoutError, Exception) as e:
        print(f"  C reference benchmark TIMEOUT or ERROR: {e}")
        c_time = None

    # Benchmark Triton (GPU, with separate timeout handling)
    try:
        print(f"Warming up Triton implementation ({num_warmup} iterations)...")
        start_time = time.perf_counter()
        for i in range(num_warmup):
            if time.perf_counter() - start_time > timeout_per_section:
                raise TimeoutError("Triton warmup timeout")
            for arr in tr_tensors:
                tr_tensors[arr] = tr_tensors[arr].clone()
            tr_args = build_args(s173_triton, tr_tensors, scalars)
            s173_triton(*tr_args)
        torch.cuda.synchronize()

        print(f"Benchmarking Triton implementation ({num_iterations} iterations)...")
        torch.cuda.synchronize()
        tr_start = time.perf_counter()
        bench_start = time.perf_counter()
        for i in range(num_iterations):
            if time.perf_counter() - bench_start > timeout_per_section:
                raise TimeoutError("Triton benchmark timeout")
            for arr in tr_tensors:
                tr_tensors[arr] = tr_tensors[arr].clone()
            tr_args = build_args(s173_triton, tr_tensors, scalars)
            s173_triton(*tr_args)
        torch.cuda.synchronize()
        tr_time = (time.perf_counter() - tr_start) / num_iterations
        print(f"  Triton time: {tr_time*1000:.3f} ms")
    except (TimeoutError, Exception) as e:
        print(f"  Triton benchmark TIMEOUT or ERROR: {e}")
        tr_time = None

    # Calculate speedup (handle None cases)
    if c_time is not None and tr_time is not None and tr_time > 0:
        speedup = c_time / tr_time
    else:
        speedup = None

    print("="*70)
    if c_time is not None:
        print(f"C ref time:    {c_time*1000:8.3f} ms")
    else:
        print(f"C ref time:    TIMEOUT")
    if tr_time is not None:
        print(f"Triton time:   {tr_time*1000:8.3f} ms")
    else:
        print(f"Triton time:   TIMEOUT")
    if speedup is not None:
        print(f"Speedup:       {speedup:8.2f}x")
    else:
        print(f"Speedup:       N/A (timeout)")
    print("="*70)

    # Output machine-readable format for parsing (handle None values)
    c_time_ms = c_time * 1000 if c_time is not None else -1
    tr_time_ms = tr_time * 1000 if tr_time is not None else -1
    speedup_val = speedup if speedup is not None else -1
    print(f"BENCHMARK_RESULT:{c_time_ms:.6f},{tr_time_ms:.6f},{speedup_val:.6f}")

if __name__ == "__main__":
    try:
        benchmark()
    except Exception as e:
        print(f"Benchmark error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
