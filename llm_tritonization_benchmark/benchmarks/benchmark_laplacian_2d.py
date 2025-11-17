"""
2D Laplacian Stencil Benchmark: PyTorch Baseline vs LLM Triton

Test: 5-point stencil computation
    f[i,j] = u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]

Common in heat equation, Poisson solvers, image processing
"""

import torch
import time
import sys
sys.path.insert(0, "llm_triton")
sys.path.insert(0, "baselines")

from laplacian_2d_triton_llm import triton_laplacian_2d
from laplacian_2d_baseline import laplacian_2d_baseline, laplacian_2d_conv

print("="*80)
print("2D LAPLACIAN STENCIL BENCHMARK: PyTorch vs LLM Triton")
print("="*80)
print("\nOperation: 5-point stencil - f[i,j] = u[i-1,j] + u[i+1,j] +")
print("                                       u[i,j-1] + u[i,j+1] - 4*u[i,j]")
print("Memory pattern: 5 reads per output point")
print("-"*80)

# Test sizes (batch, height, width)
# Typical for heat equation solvers and image processing
sizes = [
    (1, 512, 512),      # Small image
    (1, 1024, 1024),    # Medium image
    (1, 2048, 2048),    # Large image
    (4, 512, 512),      # Small batch
    (16, 256, 256),     # Larger batch
]

print(f"\n{'Size (B×H×W)':<20} {'PyTorch(slice)':<18} {'PyTorch(conv)':<18} {'Triton':<15} {'Triton/PyTorch':<15}")
print("-"*80)

results = []

for batch, height, width in sizes:
    # Create input on GPU
    u = torch.randn(batch, height, width, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(5):
        _ = laplacian_2d_baseline(u)
        _ = laplacian_2d_conv(u)
        _ = triton_laplacian_2d(u)
    torch.cuda.synchronize()

    # Benchmark PyTorch baseline (slicing)
    start = time.perf_counter()
    for _ in range(50):
        f_pytorch = laplacian_2d_baseline(u)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / 50 * 1000  # ms

    # Benchmark PyTorch conv2d
    start = time.perf_counter()
    for _ in range(50):
        f_conv = laplacian_2d_conv(u)
    torch.cuda.synchronize()
    conv_time = (time.perf_counter() - start) / 50 * 1000  # ms

    # Benchmark Triton
    start = time.perf_counter()
    for _ in range(50):
        f_triton = triton_laplacian_2d(u)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 50 * 1000  # ms

    # Verify correctness against slicing baseline
    max_diff_slice = torch.max(torch.abs(f_pytorch - f_triton)).item()
    max_diff_conv = torch.max(torch.abs(f_conv - f_triton)).item()
    assert max_diff_slice < 1e-4, f"Slice: Results don't match! Max diff: {max_diff_slice}"
    assert max_diff_conv < 1e-4, f"Conv: Results don't match! Max diff: {max_diff_conv}"

    # Calculate speedup (against slicing baseline)
    speedup = pytorch_time / triton_time

    size_str = f"{batch}×{height}×{width}"
    print(f"{size_str:<20} {pytorch_time:>12.4f} ms  {conv_time:>12.4f} ms  {triton_time:>10.4f} ms  {speedup:>13.2f}x")

    results.append({
        'size': size_str,
        'pytorch_time': pytorch_time,
        'conv_time': conv_time,
        'triton_time': triton_time,
        'speedup': speedup
    })

print("="*80)
print("\n📊 SUMMARY")
print("-"*80)

import numpy as np
avg_speedup = np.mean([r['speedup'] for r in results])
print(f"\nAverage Speedup (Triton vs PyTorch slicing): {avg_speedup:.2f}x")

# Also compare against conv2d (the highly optimized baseline)
conv_speedups = [r['conv_time'] / r['triton_time'] for r in results]
avg_conv_speedup = np.mean(conv_speedups)
print(f"Average Speedup (Triton vs PyTorch conv2d):  {avg_conv_speedup:.2f}x")

print("\nAssessment:")
if avg_speedup >= 1.5:
    print("✅ EXCELLENT - Triton significantly faster than naive PyTorch")
elif avg_speedup >= 1.1:
    print("✓✓ GOOD - Triton moderately faster than naive PyTorch")
elif avg_speedup >= 0.9:
    print("✓  ADEQUATE - Triton matches PyTorch performance")
else:
    print("⚠  PyTorch is faster - Triton has overhead for this operation")

print("\nNote:")
print("  Stencil computations like Laplacian are memory-bound operations.")
print("  PyTorch conv2d is highly optimized (uses cuDNN under the hood).")
print("  Custom Triton kernels may struggle to beat cuDNN for simple stencils.")
print("  Triton excels when fusing stencils with other operations (e.g., heat")
print("  equation time-stepping with Laplacian + scaling).")
