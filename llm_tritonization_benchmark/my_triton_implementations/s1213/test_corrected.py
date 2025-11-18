#!/usr/bin/env python3
"""
Test corrected s1213 vs buggy baseline/triton
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

from baselines.s1213_baseline import s1213_pytorch as s1213_buggy
from baselines.s1213_baseline_correct import s1213_pytorch as s1213_correct
from llm_triton.s1213_triton_llm import s1213_triton

print("="*80)
print("s1213 Correctness Analysis")
print("="*80)

# Small test to show the difference clearly
N = 10
torch.manual_seed(42)
a = torch.randn(N, device='cuda', dtype=torch.float32)
b = torch.randn(N, device='cuda', dtype=torch.float32)
c = torch.randn(N, device='cuda', dtype=torch.float32)
d = torch.randn(N, device='cuda', dtype=torch.float32)

print("\nInitial arrays:")
print(f"a = {a.cpu().numpy()}")
print(f"b = {b.cpu().numpy()}")
print(f"c = {c.cpu().numpy()}")
print(f"d = {d.cpu().numpy()}")

# Run corrected version (true sequential C semantics)
a_correct, b_correct = s1213_correct(a.clone(), b.clone(), c.clone(), d.clone())

# Run buggy baseline
a_buggy, b_buggy = s1213_buggy(a.clone(), b.clone(), c.clone(), d.clone())

# Run Triton
a_triton, b_triton = s1213_triton(a.clone(), b.clone(), c.clone(), d.clone())

print("\n" + "="*80)
print("Results Comparison:")
print("="*80)

print("\nArray a (modified by both implementations):")
print(f"Correct:  {a_correct.cpu().numpy()}")
print(f"Buggy BL: {a_buggy.cpu().numpy()}")
print(f"Triton:   {a_triton.cpu().numpy()}")

print("\nArray b (only backward dependency, should match):")
print(f"Correct:  {b_correct.cpu().numpy()}")
print(f"Buggy BL: {b_buggy.cpu().numpy()}")
print(f"Triton:   {b_triton.cpu().numpy()}")

# Compute errors
error_buggy_a = torch.max(torch.abs(a_buggy - a_correct)).item()
error_buggy_b = torch.max(torch.abs(b_buggy - b_correct)).item()
error_triton_a = torch.max(torch.abs(a_triton - a_correct)).item()
error_triton_b = torch.max(torch.abs(b_triton - b_correct)).item()

print("\n" + "="*80)
print("Error Analysis:")
print("="*80)
print(f"\nBuggy Baseline vs Correct:")
print(f"  Array a error: {error_buggy_a:.6f}")
print(f"  Array b error: {error_buggy_b:.6f}")

print(f"\nTriton vs Correct:")
print(f"  Array a error: {error_triton_a:.6f}")
print(f"  Array b error: {error_triton_b:.6f}")

print(f"\nTriton vs Buggy Baseline:")
error_triton_buggy_a = torch.max(torch.abs(a_triton - a_buggy)).item()
error_triton_buggy_b = torch.max(torch.abs(b_triton - b_buggy)).item()
print(f"  Array a error: {error_triton_buggy_a:.6f}")
print(f"  Array b error: {error_triton_buggy_b:.6f}")

print("\n" + "="*80)
print("Conclusion:")
print("="*80)

if error_buggy_a > 1e-5:
    print("❌ Buggy baseline DOES NOT match sequential C (error in 'a')")
else:
    print("✓ Buggy baseline matches sequential C")

if error_triton_a > 1e-5:
    print("❌ Triton DOES NOT match sequential C (error in 'a')")
else:
    print("✓ Triton matches sequential C")

if error_triton_buggy_a < 1e-5 and error_triton_buggy_b < 1e-5:
    print("⚠️  Triton MATCHES buggy baseline perfectly - BOTH ARE WRONG!")
else:
    print("✓ Triton differs from buggy baseline")

print("\n" + "="*80)
print("Key Insight:")
print("="*80)
print("""
The buggy implementations compute:
  Step 1: a[1:-1] = b_original[:-2] + c[1:-1]  (all at once)
  Step 2: b[1:-1] = a_original[2:] * d[1:-1]   (all at once)

But sequential C does:
  for i in 1..n-2:
    a[i] = b[i-1] + c[i]  (uses b[i-1] which may be modified!)
    b[i] = a[i+1] * d[i]  (uses original a[i+1])

The difference:
- Iteration i=2 should use b[1] which was MODIFIED by iteration i=1
- But buggy versions use the ORIGINAL b[1]

This creates a dependency chain through array 'b' that requires sequential execution!
""")
print("="*80)
