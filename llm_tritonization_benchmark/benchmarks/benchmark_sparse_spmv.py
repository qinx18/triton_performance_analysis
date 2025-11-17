#!/usr/bin/env python3
"""
Benchmark: Sparse Matrix-Vector Multiplication (SpMV)
Comparison: Baseline (PyTorch) vs LLM Triton

Note: No expert Triton implementation found for SpMV in CSR format.
This is a 2-way comparison.
"""

import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from baselines.sparse_spmv_baseline import create_random_sparse_matrix, spmv_baseline, get_csr_arrays
from llm_triton.sparse_spmv_triton_llm import spmv_triton

def benchmark_spmv(matrix_size, sparsity, warmup=3, iterations=10):
    """
    Benchmark Sparse Matrix-Vector Multiplication

    Args:
        matrix_size: (rows, cols) tuple
        sparsity: Fraction of zero elements (e.g., 0.95 = 95% sparse)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        dict: Results including timings and correctness
    """
    M, N = matrix_size
    print(f"\n{'='*60}")
    print(f"Testing {M}x{N} matrix with {sparsity*100:.1f}% sparsity")
    print(f"{'='*60}")

    # Create sparse matrix in CSR format
    A_csr = create_random_sparse_matrix(M, N, sparsity=sparsity)

    # Get CSR arrays
    values, col_indices, row_ptr, num_rows, num_cols = get_csr_arrays(A_csr)
    nnz = len(values)

    print(f"Non-zero elements: {nnz:,}")
    print(f"Storage efficiency: {nnz/(M*N)*100:.2f}% of dense")

    # Create dense vector
    x = torch.randn(N, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        _ = spmv_baseline(A_csr, x)
        _ = spmv_triton(A_csr, x)

    torch.cuda.synchronize()

    # Benchmark Baseline
    baseline_times = []
    y_baseline = None
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = spmv_baseline(A_csr, x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        baseline_times.append((end - start) * 1000)  # Convert to ms
        if i == 0:
            y_baseline = result

    baseline_mean = sum(baseline_times) / len(baseline_times)
    print(f"\n{'Baseline (PyTorch)':.<50} {baseline_mean:>8.3f} ms")

    # Benchmark LLM Triton
    llm_times = []
    y_llm = None
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = spmv_triton(A_csr, x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        llm_times.append((end - start) * 1000)
        if i == 0:
            y_llm = result

    llm_mean = sum(llm_times) / len(llm_times)
    print(f"{'LLM Triton':.<50} {llm_mean:>8.3f} ms")

    # Performance comparison
    speedup = baseline_mean / llm_mean
    print(f"\n{'Speedup (LLM vs Baseline)':.<50} {speedup:>8.2f}x")

    # Correctness check
    max_diff = (y_baseline - y_llm).abs().max().item()
    is_correct = max_diff < 1e-3

    print(f"\n{'Correctness Check':.<50}")
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Correct: {is_correct}")

    return {
        'matrix_size': matrix_size,
        'sparsity': sparsity,
        'nnz': nnz,
        'baseline_time': baseline_mean,
        'llm_time': llm_mean,
        'speedup': speedup,
        'max_diff': max_diff,
        'correct': is_correct
    }

def main():
    print("="*60)
    print("Sparse Matrix-Vector Multiplication (SpMV) Benchmark")
    print("Baseline (PyTorch) vs LLM Triton")
    print("="*60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Test configurations: (size, sparsity)
    test_configs = [
        ((4096, 4096), 0.95),    # 4K x 4K, 95% sparse
        ((8192, 8192), 0.95),    # 8K x 8K, 95% sparse
        ((16384, 16384), 0.95),  # 16K x 16K, 95% sparse
        ((4096, 4096), 0.99),    # 4K x 4K, 99% sparse (very sparse)
    ]

    results = []

    for matrix_size, sparsity in test_configs:
        try:
            result = benchmark_spmv(matrix_size, sparsity)
            results.append(result)
        except Exception as e:
            print(f"\nERROR testing {matrix_size} with {sparsity*100:.1f}% sparsity: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Size':<15} {'Sparsity':<10} {'Baseline (ms)':<15} {'LLM (ms)':<15} {'Speedup':<10} {'Correct':<10}")
    print("-"*80)

    for r in results:
        size_str = f"{r['matrix_size'][0]}x{r['matrix_size'][1]}"
        sparsity_str = f"{r['sparsity']*100:.1f}%"
        correct_str = "✓" if r['correct'] else "✗"
        print(f"{size_str:<15} {sparsity_str:<10} {r['baseline_time']:<15.3f} {r['llm_time']:<15.3f} "
              f"{r['speedup']:<10.2f}x {correct_str:<10}")

    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print(f"\nAverage Speedup: {avg_speedup:.2f}x")

        # Overall verdict
        print("\n" + "="*60)
        print("VERDICT")
        print("="*60)

        all_correct = all(r['correct'] for r in results)
        if not all_correct:
            print("❌ FAILED - Correctness issues detected")
        elif avg_speedup >= 1.0:
            print(f"✅ SUCCESS - LLM Triton is {avg_speedup:.2f}x faster than baseline")
        else:
            print(f"⚠️  SLOWER - LLM Triton is {1/avg_speedup:.2f}x slower than baseline")

if __name__ == "__main__":
    main()
