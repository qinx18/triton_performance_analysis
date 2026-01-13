#!/usr/bin/env python3
"""
Performance Benchmark for s1119
"""
import sys
import time
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s1119_baseline import s1119_pytorch
    from test16.llm_triton.s1119.attempt1 import s1119_triton
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
    N = 256
    num_warmup = 10
    num_iterations = 100

    print("="*70)
    print(f"Performance Benchmark: s1119")
    print(f"Array size: N={N}")
    print("="*70)

    # Initialize arrays
    aa = torch.randn(N + 10, N + 10, device='cuda', dtype=torch.float32)
    bb = torch.randn(N + 10, N + 10, device='cuda', dtype=torch.float32)
    iterations = 1

    pt_tensors = {"aa": aa, "bb": bb}
    tr_tensors = {"aa": aa.clone(), "bb": bb.clone()}
    scalars = {"iterations": iterations}

    pt_args = build_args(s1119_pytorch, pt_tensors, scalars)
    tr_args = build_args(s1119_triton, tr_tensors, scalars)

    # Warmup PyTorch
    print(f"Warming up PyTorch baseline ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        for arr in pt_tensors:
            pt_tensors[arr] = pt_tensors[arr].clone()
        pt_args = build_args(s1119_pytorch, pt_tensors, scalars)
        s1119_pytorch(*pt_args)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    print(f"Benchmarking PyTorch baseline ({num_iterations} iterations)...")
    torch.cuda.synchronize()
    pt_start = time.perf_counter()
    for _ in range(num_iterations):
        for arr in pt_tensors:
            pt_tensors[arr] = pt_tensors[arr].clone()
        pt_args = build_args(s1119_pytorch, pt_tensors, scalars)
        s1119_pytorch(*pt_args)
    torch.cuda.synchronize()
    pt_time = (time.perf_counter() - pt_start) / num_iterations

    # Warmup Triton
    print(f"Warming up Triton implementation ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        for arr in tr_tensors:
            tr_tensors[arr] = tr_tensors[arr].clone()
        tr_args = build_args(s1119_triton, tr_tensors, scalars)
        s1119_triton(*tr_args)
    torch.cuda.synchronize()

    # Benchmark Triton
    print(f"Benchmarking Triton implementation ({num_iterations} iterations)...")
    torch.cuda.synchronize()
    tr_start = time.perf_counter()
    for _ in range(num_iterations):
        for arr in tr_tensors:
            tr_tensors[arr] = tr_tensors[arr].clone()
        tr_args = build_args(s1119_triton, tr_tensors, scalars)
        s1119_triton(*tr_args)
    torch.cuda.synchronize()
    tr_time = (time.perf_counter() - tr_start) / num_iterations

    speedup = pt_time / tr_time if tr_time > 0 else 0

    print("="*70)
    print(f"PyTorch time:  {pt_time*1000:8.3f} ms")
    print(f"Triton time:   {tr_time*1000:8.3f} ms")
    print(f"Speedup:       {speedup:8.2f}x")
    print("="*70)

    # Output machine-readable format for parsing
    print(f"BENCHMARK_RESULT:{pt_time*1000:.6f},{tr_time*1000:.6f},{speedup:.6f}")

if __name__ == "__main__":
    try:
        benchmark()
    except Exception as e:
        print(f"Benchmark error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
