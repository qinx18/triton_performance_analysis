#!/usr/bin/env python3
"""
Test corrected s211 implementations
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

from baselines.s211_baseline import s211_pytorch as s211_buggy
from baselines.s211_baseline_correct import s211_pytorch as s211_correct
from llm_triton.s211_triton_llm import s211_triton as s211_triton_buggy
from llm_triton.s211_triton_correct import s211_triton as s211_triton_correct

def s211_sequential_c(a, b, c, d, e):
    """
    True sequential C execution:
    for (int i = 1; i < n-1; i++) {
        a[i] = b[i-1] + c[i] * d[i];
        b[i] = b[i+1] - e[i] * d[i];
    }
    """
    a = a.clone()
    b = b.clone()
    c = c.clone()
    d = d.clone()
    e = e.clone()

    n = a.shape[0]

    for i in range(1, n - 1):
        a[i] = b[i - 1] + c[i] * d[i]
        b[i] = b[i + 1] - e[i] * d[i]

    return a, b

print("="*80)
print("s211 Corrected Implementation Test")
print("="*80)

# Test with small array to show the difference
N = 10
torch.manual_seed(42)
a = torch.randn(N, device='cuda', dtype=torch.float32)
b = torch.randn(N, device='cuda', dtype=torch.float32)
c = torch.randn(N, device='cuda', dtype=torch.float32)
d = torch.randn(N, device='cuda', dtype=torch.float32)
e = torch.randn(N, device='cuda', dtype=torch.float32)

print("\nInitial arrays:")
print(f"a = {a.cpu().numpy()}")
print(f"b = {b.cpu().numpy()}")

# Run all versions
a_seq, b_seq = s211_sequential_c(a.clone(), b.clone(), c.clone(), d.clone(), e.clone())
a_buggy, b_buggy = s211_buggy(a.clone(), b.clone(), c.clone(), d.clone(), e.clone())
a_correct, b_correct = s211_correct(a.clone(), b.clone(), c.clone(), d.clone(), e.clone())
a_triton_buggy, b_triton_buggy = s211_triton_buggy(a.clone(), b.clone(), c.clone(), d.clone(), e.clone())
a_triton_correct, b_triton_correct = s211_triton_correct(a.clone(), b.clone(), c.clone(), d.clone(), e.clone())

print("\n" + "="*80)
print("Results:")
print("="*80)
print("\nArray a:")
print(f"Sequential C:      {a_seq.cpu().numpy()}")
print(f"Buggy Baseline:    {a_buggy.cpu().numpy()}")
print(f"Corrected Baseline:{a_correct.cpu().numpy()}")
print(f"Buggy Triton:      {a_triton_buggy.cpu().numpy()}")
print(f"Corrected Triton:  {a_triton_correct.cpu().numpy()}")

print("\nArray b:")
print(f"Sequential C:      {b_seq.cpu().numpy()}")
print(f"Buggy Baseline:    {b_buggy.cpu().numpy()}")
print(f"Corrected Baseline:{b_correct.cpu().numpy()}")
print(f"Buggy Triton:      {b_triton_buggy.cpu().numpy()}")
print(f"Corrected Triton:  {b_triton_correct.cpu().numpy()}")

# Compute errors
print("\n" + "="*80)
print("Error Analysis:")
print("="*80)

error_buggy_a = torch.max(torch.abs(a_buggy - a_seq)).item()
error_buggy_b = torch.max(torch.abs(b_buggy - b_seq)).item()
error_correct_a = torch.max(torch.abs(a_correct - a_seq)).item()
error_correct_b = torch.max(torch.abs(b_correct - b_seq)).item()
error_triton_buggy_a = torch.max(torch.abs(a_triton_buggy - a_seq)).item()
error_triton_buggy_b = torch.max(torch.abs(b_triton_buggy - b_seq)).item()
error_triton_correct_a = torch.max(torch.abs(a_triton_correct - a_seq)).item()
error_triton_correct_b = torch.max(torch.abs(b_triton_correct - b_seq)).item()

print(f"\nBuggy Baseline vs Sequential C:")
print(f"  Array a error: {error_buggy_a:.6f}")
print(f"  Array b error: {error_buggy_b:.6f}")
if error_buggy_a > 1e-4 or error_buggy_b > 1e-4:
    print(f"  ❌ WRONG")
else:
    print(f"  ✓ Correct")

print(f"\nCorrected Baseline vs Sequential C:")
print(f"  Array a error: {error_correct_a:.6f}")
print(f"  Array b error: {error_correct_b:.6f}")
if error_correct_a > 1e-4 or error_correct_b > 1e-4:
    print(f"  ❌ WRONG")
else:
    print(f"  ✓ Correct")

print(f"\nBuggy Triton vs Sequential C:")
print(f"  Array a error: {error_triton_buggy_a:.6f}")
print(f"  Array b error: {error_triton_buggy_b:.6f}")
if error_triton_buggy_a > 1e-4 or error_triton_buggy_b > 1e-4:
    print(f"  ❌ WRONG")
else:
    print(f"  ✓ Correct")

print(f"\nCorrected Triton vs Sequential C:")
print(f"  Array a error: {error_triton_correct_a:.6f}")
print(f"  Array b error: {error_triton_correct_b:.6f}")
if error_triton_correct_a > 1e-4 or error_triton_correct_b > 1e-4:
    print(f"  ❌ WRONG")
else:
    print(f"  ✓ Correct")

print("\n" + "="*80)
print("Testing larger sizes...")
print("="*80)

for N in [100, 1000, 10000]:
    torch.manual_seed(42)
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    c = torch.randn(N, device='cuda', dtype=torch.float32)
    d = torch.randn(N, device='cuda', dtype=torch.float32)
    e = torch.randn(N, device='cuda', dtype=torch.float32)

    a_seq, b_seq = s211_sequential_c(a.clone(), b.clone(), c.clone(), d.clone(), e.clone())
    a_correct, b_correct = s211_correct(a.clone(), b.clone(), c.clone(), d.clone(), e.clone())
    a_triton_correct, b_triton_correct = s211_triton_correct(a.clone(), b.clone(), c.clone(), d.clone(), e.clone())

    error_baseline_a = torch.max(torch.abs(a_correct - a_seq)).item()
    error_baseline_b = torch.max(torch.abs(b_correct - b_seq)).item()
    error_triton_a = torch.max(torch.abs(a_triton_correct - a_seq)).item()
    error_triton_b = torch.max(torch.abs(b_triton_correct - b_seq)).item()

    passed = (error_baseline_a < 1e-4 and error_baseline_b < 1e-4 and
              error_triton_a < 1e-4 and error_triton_b < 1e-4)

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"N={N:>6}: {status}  (baseline_err={max(error_baseline_a, error_baseline_b):.2e}, triton_err={max(error_triton_a, error_triton_b):.2e})")

print("="*80)
