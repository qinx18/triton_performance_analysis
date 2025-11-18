#!/usr/bin/env python3
"""
Test s1213 for race condition by comparing against true sequential C execution
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

from baselines.s1213_baseline import s1213_pytorch
from llm_triton.s1213_triton_llm import s1213_triton

def s1213_sequential_c(a, b, c, d):
    """
    True sequential C execution:
    for (int i = 1; i < LEN_1D-1; i++) {
        a[i] = b[i-1]+c[i];   // Writes to a[i]
        b[i] = a[i+1]*d[i];   // Reads ORIGINAL a[i+1] (not yet modified)
    }
    """
    a = a.clone()
    b = b.clone()
    c = c.clone()
    d = d.clone()

    n = a.shape[0]

    # Process sequentially
    for i in range(1, n - 1):
        # Save original a[i+1] before modifying a[i]
        a_next_val = a[i + 1].item()

        # Line 1: a[i] = b[i-1] + c[i]
        a[i] = b[i - 1] + c[i]

        # Line 2: b[i] = a[i+1] * d[i] (uses ORIGINAL a[i+1])
        b[i] = a_next_val * d[i]

    return a, b

def test_race_condition():
    """Test if Triton matches true sequential C execution"""
    test_sizes = [100, 1000, 10000]
    all_passed = True

    print("="*80)
    print("s1213 Race Condition Analysis")
    print("="*80)
    print("\nC Algorithm:")
    print("  for (i = 1; i < n-1; i++) {")
    print("    a[i] = b[i-1] + c[i];   // Writes a[i]")
    print("    b[i] = a[i+1] * d[i];   // Reads a[i+1] (should be ORIGINAL)")
    print("  }")
    print("\nPotential race: Thread i+1 writes a[i+1] before thread i reads it")
    print("="*80)

    for N in test_sizes:
        print(f"\nTesting N={N}...")

        # Initialize arrays with specific pattern to expose race conditions
        torch.manual_seed(42)
        a = torch.randn(N, device='cuda', dtype=torch.float32)
        b = torch.randn(N, device='cuda', dtype=torch.float32)
        c = torch.randn(N, device='cuda', dtype=torch.float32)
        d = torch.randn(N, device='cuda', dtype=torch.float32)

        # Run sequential C version (ground truth)
        a_seq, b_seq = s1213_sequential_c(a.clone(), b.clone(), c.clone(), d.clone())

        # Run PyTorch baseline
        a_baseline, b_baseline = s1213_pytorch(a.clone(), b.clone(), c.clone(), d.clone())

        # Run Triton
        a_triton, b_triton = s1213_triton(a.clone(), b.clone(), c.clone(), d.clone())

        # Compare baseline vs sequential C
        error_baseline_a = torch.max(torch.abs(a_baseline - a_seq)).item()
        error_baseline_b = torch.max(torch.abs(b_baseline - b_seq)).item()

        # Compare Triton vs sequential C
        error_triton_a = torch.max(torch.abs(a_triton - a_seq)).item()
        error_triton_b = torch.max(torch.abs(b_triton - b_seq)).item()

        # Compare Triton vs baseline
        error_triton_baseline_a = torch.max(torch.abs(a_triton - a_baseline)).item()
        error_triton_baseline_b = torch.max(torch.abs(b_triton - b_baseline)).item()

        print(f"  Baseline vs Sequential C: a_err={error_baseline_a:.2e}, b_err={error_baseline_b:.2e}")
        print(f"  Triton   vs Sequential C: a_err={error_triton_a:.2e}, b_err={error_triton_b:.2e}")
        print(f"  Triton   vs Baseline:     a_err={error_triton_baseline_a:.2e}, b_err={error_triton_baseline_b:.2e}")

        # Check if baseline matches sequential C
        if error_baseline_a > 1e-5 or error_baseline_b > 1e-5:
            print(f"  ⚠️  BASELINE DOESN'T MATCH SEQUENTIAL C!")
            all_passed = False
        else:
            print(f"  ✓ Baseline matches sequential C")

        # Check if Triton matches sequential C
        if error_triton_a > 1e-5 or error_triton_b > 1e-5:
            print(f"  ❌ TRITON DOESN'T MATCH SEQUENTIAL C - RACE CONDITION!")
            all_passed = False
        else:
            print(f"  ✓ Triton matches sequential C (no race detected)")

        # Check if Triton matches baseline
        if error_triton_baseline_a > 1e-5 or error_triton_baseline_b > 1e-5:
            print(f"  ⚠️  TRITON DOESN'T MATCH BASELINE!")
            all_passed = False

    print("="*80)
    if all_passed:
        print("✅ No race condition detected - all implementations match!")
    else:
        print("❌ Race condition or implementation bug detected!")
    print("="*80)

    return all_passed

if __name__ == "__main__":
    success = test_race_condition()
    sys.exit(0 if success else 1)
