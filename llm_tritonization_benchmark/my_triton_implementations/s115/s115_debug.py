#!/usr/bin/env python3
"""
Detailed debugging for s115 to find where Triton diverges from PyTorch
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from baselines.s115_baseline import s115_pytorch
from llm_triton.s115_triton_llm import s115_triton

def test_s115_detailed(N):
    """Test s115 with detailed iteration-by-iteration comparison"""
    print("="*80)
    print(f"DETAILED S115 DEBUG: N={N}")
    print("="*80)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Initialize arrays
    a_init = torch.randn(N, device='cuda', dtype=torch.float32)
    aa_init = torch.randn(N, N, device='cuda', dtype=torch.float32)

    print(f"\nInitial values:")
    print(f"a[:10] = {a_init[:min(10, N)]}")
    print(f"aa[0,:10] = {aa_init[0, :min(10, N)]}")

    # Run PyTorch baseline
    print("\n" + "-"*80)
    print("PYTORCH BASELINE")
    print("-"*80)
    a_pytorch = a_init.clone()
    aa_pytorch = aa_init.clone()

    # Step through manually to see intermediate states
    for j in range(min(5, N)):  # Only show first 5 iterations
        a_before = a_pytorch.clone()
        for i in range(j + 1, N):
            a_pytorch[i] -= aa_pytorch[j, i] * a_pytorch[j]

        max_change = torch.max(torch.abs(a_pytorch - a_before)).item()
        print(f"j={j}: max_change={max_change:.6e}, a[{j}]={a_pytorch[j]:.6f}, a[{min(j+1, N-1)}]={a_pytorch[min(j+1, N-1)]:.6f}")

    # Complete the rest without printing
    for j in range(5, N):
        for i in range(j + 1, N):
            a_pytorch[i] -= aa_pytorch[j, i] * a_pytorch[j]

    print(f"\nFinal PyTorch a[:10] = {a_pytorch[:min(10, N)]}")

    # Run Triton
    print("\n" + "-"*80)
    print("TRITON IMPLEMENTATION")
    print("-"*80)
    a_triton = a_init.clone()
    aa_triton = aa_init.clone()

    result_triton = s115_triton(a_triton, aa_triton)

    print(f"Final Triton a[:10] = {result_triton[:min(10, N)]}")

    # Compare
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    diff = a_pytorch - result_triton
    abs_diff = torch.abs(diff)
    max_error = torch.max(abs_diff).item()
    mean_error = torch.mean(abs_diff).item()

    print(f"Max error: {max_error:.6e}")
    print(f"Mean error: {mean_error:.6e}")

    # Show where largest errors occur
    if max_error > 1e-5:
        max_idx = torch.argmax(abs_diff).item()
        print(f"\nLargest error at index {max_idx}:")
        print(f"  PyTorch: {a_pytorch[max_idx]:.10f}")
        print(f"  Triton:  {result_triton[max_idx]:.10f}")
        print(f"  Diff:    {diff[max_idx]:.10e}")

        # Show top 10 errors
        top_errors, top_indices = torch.topk(abs_diff, min(10, N))
        print(f"\nTop 10 errors:")
        for i, (err, idx) in enumerate(zip(top_errors, top_indices)):
            print(f"  {i+1}. idx={idx:4d}: error={err:.6e}, pytorch={a_pytorch[idx]:.6f}, triton={result_triton[idx]:.6f}")

    return max_error < 1e-4


# Test with increasing sizes to find failure point
print("\n" + "="*80)
print("TESTING MULTIPLE SIZES")
print("="*80)

test_sizes = [5, 10, 20, 30, 50, 100]

for N in test_sizes:
    print(f"\n{'='*80}")
    print(f"N = {N}")
    print('='*80)

    torch.manual_seed(42)
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    aa = torch.randn(N, N, device='cuda', dtype=torch.float32)

    pytorch_result = s115_pytorch(a.clone(), aa.clone())
    triton_result = s115_triton(a.clone(), aa.clone())

    max_error = torch.max(torch.abs(pytorch_result - triton_result)).item()

    if max_error < 1e-4:
        print(f"✓ PASS: max_error={max_error:.2e}")
    else:
        print(f"✗ FAIL: max_error={max_error:.2e}")
        # Run detailed analysis for this size
        test_s115_detailed(N)
        break  # Stop at first failure
