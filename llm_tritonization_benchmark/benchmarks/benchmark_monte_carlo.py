#!/usr/bin/env python3
"""
Benchmark: Monte Carlo Pi Estimation
Comparison: Baseline (PyTorch) vs LLM Triton

Note: No expert Triton implementation found for Monte Carlo methods.
This is a 2-way comparison.
"""

import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from baselines.monte_carlo_baseline import monte_carlo_pi_baseline
from llm_triton.monte_carlo_triton_llm import monte_carlo_pi_triton

def benchmark_monte_carlo(n_samples, warmup=3, iterations=10):
    """
    Benchmark Monte Carlo Pi estimation

    Args:
        n_samples: Number of random samples for Pi estimation
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        dict: Results including timings and accuracy
    """
    print(f"\n{'='*60}")
    print(f"Testing with {n_samples:,} samples")
    print(f"{'='*60}")

    # True value of Pi for accuracy comparison
    true_pi = 3.141592653589793

    # Warmup
    for _ in range(warmup):
        _ = monte_carlo_pi_baseline(n_samples)
        _ = monte_carlo_pi_triton(n_samples)

    torch.cuda.synchronize()

    # Benchmark Baseline
    baseline_times = []
    baseline_pi = None
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = monte_carlo_pi_baseline(n_samples)
        torch.cuda.synchronize()
        end = time.perf_counter()
        baseline_times.append((end - start) * 1000)  # Convert to ms
        if i == 0:
            baseline_pi = result

    baseline_mean = sum(baseline_times) / len(baseline_times)
    baseline_error = abs(baseline_pi - true_pi)

    print(f"\n{'Baseline (PyTorch)':.<50} {baseline_mean:>8.3f} ms")
    print(f"  → Pi estimate: {baseline_pi:.10f} (error: {baseline_error:.6f})")

    # Benchmark LLM Triton
    llm_times = []
    llm_pi = None
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = monte_carlo_pi_triton(n_samples)
        torch.cuda.synchronize()
        end = time.perf_counter()
        llm_times.append((end - start) * 1000)
        if i == 0:
            llm_pi = result

    llm_mean = sum(llm_times) / len(llm_times)
    llm_error = abs(llm_pi - true_pi)

    print(f"{'LLM Triton':.<50} {llm_mean:>8.3f} ms")
    print(f"  → Pi estimate: {llm_pi:.10f} (error: {llm_error:.6f})")

    # Performance comparison
    speedup = baseline_mean / llm_mean
    print(f"\n{'Speedup (LLM vs Baseline)':.<50} {speedup:>8.2f}x")

    # Accuracy comparison
    print(f"\n{'Accuracy Comparison':.<50}")
    print(f"  Baseline error: {baseline_error:.8f}")
    print(f"  LLM error:      {llm_error:.8f}")
    accuracy_ratio = baseline_error / llm_error if llm_error > 0 else float('inf')
    print(f"  Error ratio (baseline/LLM): {accuracy_ratio:.2f}x")

    return {
        'n_samples': n_samples,
        'baseline_time': baseline_mean,
        'llm_time': llm_mean,
        'speedup': speedup,
        'baseline_pi': baseline_pi,
        'llm_pi': llm_pi,
        'baseline_error': baseline_error,
        'llm_error': llm_error
    }

def main():
    print("="*60)
    print("Monte Carlo Pi Estimation Benchmark")
    print("Baseline (PyTorch) vs LLM Triton")
    print("="*60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Test sizes - Monte Carlo needs large samples for accuracy
    test_sizes = [
        1_000_000,      # 1M samples
        10_000_000,     # 10M samples
        100_000_000,    # 100M samples
    ]

    results = []

    for n_samples in test_sizes:
        try:
            result = benchmark_monte_carlo(n_samples)
            results.append(result)
        except Exception as e:
            print(f"\nERROR testing {n_samples:,} samples: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Samples':<15} {'Baseline (ms)':<15} {'LLM (ms)':<15} {'Speedup':<10} {'Accuracy':<10}")
    print("-"*60)

    for r in results:
        accuracy_status = "✓" if r['llm_error'] < 0.01 else "✗"
        print(f"{r['n_samples']:<15,} {r['baseline_time']:<15.3f} {r['llm_time']:<15.3f} "
              f"{r['speedup']:<10.2f}x {accuracy_status:<10}")

    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print(f"\nAverage Speedup: {avg_speedup:.2f}x")

        # Overall verdict
        print("\n" + "="*60)
        print("VERDICT")
        print("="*60)

        all_correct = all(r['llm_error'] < 0.01 for r in results)
        if not all_correct:
            print("❌ FAILED - Accuracy issues detected")
        elif avg_speedup >= 1.0:
            print(f"✅ SUCCESS - LLM Triton is {avg_speedup:.2f}x faster than baseline")
        else:
            print(f"⚠️  SLOWER - LLM Triton is {1/avg_speedup:.2f}x slower than baseline")

if __name__ == "__main__":
    main()
