"""
s000 Benchmark: PyTorch Baseline vs LLM Triton

Test kernel: a[i] = b[i] + 1 (simple vector add with scalar)
From TSVC (Test Suite for Vectorizing Compilers)
"""

import torch
import time
import sys
sys.path.insert(0, "llm_triton")
sys.path.insert(0, "baselines")

from s000_triton_llm import s000_triton
from s000_baseline import s000_baseline

print("="*80)
print("s000 BENCHMARK: PyTorch Baseline vs LLM Triton")
print("="*80)
print("\nOperation: a[i] = b[i] + 1")
print("Source: TSVC (Test Suite for Vectorizing Compilers)")
print("-"*80)

# Test sizes (similar to TSVC default)
sizes = [32000, 64000, 128000, 256000, 512000]

print(f"\n{'Size':<12} {'PyTorch':<15} {'Triton':<15} {'Speedup':<12}")
print("-"*80)

results = []

for size in sizes:
    # Create input on GPU
    b = torch.randn(size, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = s000_baseline(b)
        _ = s000_triton(b)
    torch.cuda.synchronize()

    # Benchmark PyTorch baseline
    start = time.perf_counter()
    for _ in range(100):
        a_pytorch = s000_baseline(b)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / 100 * 1000  # ms

    # Benchmark Triton
    start = time.perf_counter()
    for _ in range(100):
        a_triton = s000_triton(b)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 100 * 1000  # ms

    # Verify correctness
    max_diff = torch.max(torch.abs(a_pytorch - a_triton)).item()
    assert max_diff < 1e-5, f"Results don't match! Max diff: {max_diff}"

    # Calculate speedup
    speedup = pytorch_time / triton_time

    print(f"{size:<12} {pytorch_time:>10.4f} ms  {triton_time:>10.4f} ms  {speedup:>10.2f}x")

    results.append({
        'size': size,
        'pytorch_time': pytorch_time,
        'triton_time': triton_time,
        'speedup': speedup
    })

print("="*80)
print("\n📊 SUMMARY")
print("-"*80)

import numpy as np
avg_speedup = np.mean([r['speedup'] for r in results])
print(f"\nAverage Speedup: {avg_speedup:.2f}x")

if avg_speedup >= 1.5:
    print("✅ EXCELLENT - Triton significantly faster than PyTorch")
elif avg_speedup >= 1.1:
    print("✓✓ GOOD - Triton moderately faster than PyTorch")
elif avg_speedup >= 0.9:
    print("✓  ADEQUATE - Triton matches PyTorch performance")
else:
    print("⚠  PyTorch is faster - Triton has overhead for this simple operation")

print("\nNote:")
print("  For simple element-wise operations like s000 (a[i] = b[i] + 1),")
print("  PyTorch built-ins are highly optimized. Triton may have kernel")
print("  launch overhead that dominates for such trivial operations.")
print("  Triton shines for fused operations with multiple memory passes.")
