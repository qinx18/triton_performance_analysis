#!/usr/bin/env python3
"""
Performance Benchmark for s000
"""
import sys
import time
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s000_baseline import s000_pytorch
    from test16.llm_triton.s000.attempt1 import s000_triton
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
    print(f"Performance Benchmark: s000")
    print(f"Array size: N={N}")
    print("="*70)

    # Initialize arrays
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    iterations = 1

    pt_tensors = {"a": a, "b": b}
    tr_tensors = {"a": a.clone(), "b": b.clone()}
    scalars = {"iterations": iterations}

    pt_args = build_args(s000_pytorch, pt_tensors, scalars)
    tr_args = build_args(s000_triton, tr_tensors, scalars)

    pt_time = None
    tr_time = None

    # Benchmark PyTorch (with separate timeout handling)
    try:
        print(f"Warming up PyTorch baseline ({num_warmup} iterations)...")
        start_time = time.perf_counter()
        for i in range(num_warmup):
            if time.perf_counter() - start_time > timeout_per_section:
                raise TimeoutError("PyTorch warmup timeout")
            for arr in pt_tensors:
                pt_tensors[arr] = pt_tensors[arr].clone()
            pt_args = build_args(s000_pytorch, pt_tensors, scalars)
            s000_pytorch(*pt_args)
        torch.cuda.synchronize()

        print(f"Benchmarking PyTorch baseline ({num_iterations} iterations)...")
        torch.cuda.synchronize()
        pt_start = time.perf_counter()
        bench_start = time.perf_counter()
        for i in range(num_iterations):
            if time.perf_counter() - bench_start > timeout_per_section:
                raise TimeoutError("PyTorch benchmark timeout")
            for arr in pt_tensors:
                pt_tensors[arr] = pt_tensors[arr].clone()
            pt_args = build_args(s000_pytorch, pt_tensors, scalars)
            s000_pytorch(*pt_args)
        torch.cuda.synchronize()
        pt_time = (time.perf_counter() - pt_start) / num_iterations
        print(f"  PyTorch time: {pt_time*1000:.3f} ms")
    except (TimeoutError, Exception) as e:
        print(f"  PyTorch benchmark TIMEOUT or ERROR: {e}")
        pt_time = None

    # Benchmark Triton (with separate timeout handling)
    try:
        print(f"Warming up Triton implementation ({num_warmup} iterations)...")
        start_time = time.perf_counter()
        for i in range(num_warmup):
            if time.perf_counter() - start_time > timeout_per_section:
                raise TimeoutError("Triton warmup timeout")
            for arr in tr_tensors:
                tr_tensors[arr] = tr_tensors[arr].clone()
            tr_args = build_args(s000_triton, tr_tensors, scalars)
            s000_triton(*tr_args)
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
            tr_args = build_args(s000_triton, tr_tensors, scalars)
            s000_triton(*tr_args)
        torch.cuda.synchronize()
        tr_time = (time.perf_counter() - tr_start) / num_iterations
        print(f"  Triton time: {tr_time*1000:.3f} ms")
    except (TimeoutError, Exception) as e:
        print(f"  Triton benchmark TIMEOUT or ERROR: {e}")
        tr_time = None

    # Calculate speedup (handle None cases)
    if pt_time is not None and tr_time is not None and tr_time > 0:
        speedup = pt_time / tr_time
    else:
        speedup = None

    print("="*70)
    if pt_time is not None:
        print(f"PyTorch time:  {pt_time*1000:8.3f} ms")
    else:
        print(f"PyTorch time:  TIMEOUT")
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
    pt_time_ms = pt_time * 1000 if pt_time is not None else -1
    tr_time_ms = tr_time * 1000 if tr_time is not None else -1
    speedup_val = speedup if speedup is not None else -1
    print(f"BENCHMARK_RESULT:{pt_time_ms:.6f},{tr_time_ms:.6f},{speedup_val:.6f}")

if __name__ == "__main__":
    try:
        benchmark()
    except Exception as e:
        print(f"Benchmark error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
