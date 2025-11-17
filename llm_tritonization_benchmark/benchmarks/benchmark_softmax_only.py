"""
Softmax 3-Way Benchmark: Baseline vs Expert Triton vs LLM Triton
"""

import torch
import time
import triton
import triton.language as tl
import sys
sys.path.insert(0, "llm_triton")
from softmax_triton_llm import triton_softmax as softmax_llm

# ========== BASELINE (from official tutorial) ==========
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch"""
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret

# ========== EXPERT TRITON (from official tutorial) ==========
@triton.jit
def softmax_kernel_expert(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                          BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax_expert(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_stages = 4 if BLOCK_SIZE > 2048 else 2
    y = torch.empty_like(x)

    softmax_kernel_expert[(n_rows, )](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
    )
    return y

# ========== BENCHMARK ==========
print("="*100)
print("SOFTMAX: 3-WAY COMPARISON (Baseline vs Expert Triton vs LLM Triton)")
print("="*100)

sizes = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 1024)]

print(f"\n{'Size':<15} {'Baseline':<12} {'Expert':<12} {'LLM':<12} {'Expert/Base':<13} {'LLM/Base':<13} {'LLM/Expert':<13}")
print("-"*100)

results = []

for m, n in sizes:
    x = torch.randn(m, n, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(5):
        naive_softmax(x)
        softmax_expert(x)
        softmax_llm(x)
    torch.cuda.synchronize()

    # Benchmark Baseline
    start = time.perf_counter()
    for _ in range(50):
        y_baseline = naive_softmax(x)
    torch.cuda.synchronize()
    base_time = (time.perf_counter() - start) / 50 * 1000

    # Benchmark Expert
    start = time.perf_counter()
    for _ in range(50):
        y_expert = softmax_expert(x)
    torch.cuda.synchronize()
    expert_time = (time.perf_counter() - start) / 50 * 1000

    # Benchmark LLM
    start = time.perf_counter()
    for _ in range(50):
        y_llm = softmax_llm(x)
    torch.cuda.synchronize()
    llm_time = (time.perf_counter() - start) / 50 * 1000

    # Speedups
    expert_vs_base = base_time / expert_time
    llm_vs_base = base_time / llm_time
    llm_vs_expert = expert_time / llm_time

    print(f"{m}x{n:<10} {base_time:>10.4f}ms {expert_time:>10.4f}ms {llm_time:>10.4f}ms "
          f"{expert_vs_base:>11.2f}x {llm_vs_base:>11.2f}x {llm_vs_expert:>11.2f}x")

    results.append({
        'size': f'{m}x{n}',
        'llm_vs_expert': llm_vs_expert
    })

print("="*100)
print("\n📊 SUMMARY")
print("-"*100)

import numpy as np
avg_llm_vs_expert = np.mean([r['llm_vs_expert'] for r in results])
print(f"\nAverage LLM/Expert ratio: {avg_llm_vs_expert:.2f}x")
print(f"LLM achieves {avg_llm_vs_expert*100:.1f}% of Expert Triton performance\n")

if avg_llm_vs_expert >= 0.9:
    print("✅ EXCELLENT - LLM matches expert performance!")
elif avg_llm_vs_expert >= 0.7:
    print("✓✓ GOOD - LLM is close to expert performance")
elif avg_llm_vs_expert >= 0.5:
    print("✓  ADEQUATE - LLM has decent performance but room for improvement")
else:
    print("✗  POOR - Significant gap between LLM and expert")

print("\nInterpretation:")
print("  - 1.0x = LLM matches expert")
print("  - 0.7-0.9x = LLM is very close to expert")
print("  - 0.5-0.7x = LLM is decent but missing optimizations")
print("  - <0.5x = LLM significantly slower than expert")
