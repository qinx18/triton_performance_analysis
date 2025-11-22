#!/usr/bin/env python3
"""
Debug s1221 to understand the bug
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

from baselines.s1221_baseline import s1221_pytorch
from llm_triton.s1221_triton_llm import s1221_triton

print("="*80)
print("s1221 Debugging")
print("="*80)
print("\nAlgorithm: b[i] = b[i-4] + a[i] for i in [4, n)")
print("\nDependency pattern (stride-4):")
print("  Chain 0: b[4], b[8], b[12], ... (each depends on previous)")
print("  Chain 1: b[5], b[9], b[13], ...")
print("  Chain 2: b[6], b[10], b[14], ...")
print("  Chain 3: b[7], b[11], b[15], ...")
print("  → 4 independent chains, but each chain is sequential")
print("="*80)

# Small test to see the pattern
N = 20
torch.manual_seed(42)
a = torch.randn(N, device='cuda', dtype=torch.float32)
b = torch.randn(N, device='cuda', dtype=torch.float32)

print(f"\nInitial arrays (N={N}):")
print(f"a = {a.cpu().numpy()}")
print(f"b = {b.cpu().numpy()}")

# Run baseline (sequential, should be correct)
b_baseline = s1221_pytorch(a.clone(), b.clone())

# Run Triton (parallel with chunking, may be wrong)
b_triton = s1221_triton(a.clone(), b.clone())

# Compute ground truth manually to show the dependency chain
b_manual = b.clone()
print("\n" + "="*80)
print("Manual computation (showing dependencies):")
print("="*80)
for i in range(4, N):
    old_val = b_manual[i].item()
    b_prev = b_manual[i - 4].item()
    a_val = a[i].item()
    b_manual[i] = b_prev + a_val
    print(f"i={i:2d}: b[{i:2d}] = b[{i-4}] + a[{i:2d}] = {b_prev:.4f} + {a_val:.4f} = {b_manual[i].item():.4f}")

print("\n" + "="*80)
print("Results comparison:")
print("="*80)
print("Index | Manual (correct) | Baseline | Triton   | Baseline err | Triton err")
print("-"*80)
for i in range(N):
    manual = b_manual[i].item()
    baseline = b_baseline[i].item()
    triton = b_triton[i].item()
    err_baseline = abs(baseline - manual)
    err_triton = abs(triton - manual)
    mark = "❌" if err_triton > 1e-4 else "  "
    print(f"{mark} {i:2d}  | {manual:14.6f}   | {baseline:8.6f} | {triton:8.6f} | {err_baseline:.2e}     | {err_triton:.2e}")

print("\n" + "="*80)
print("Analysis:")
print("="*80)

error_baseline = torch.max(torch.abs(b_baseline - b_manual)).item()
error_triton = torch.max(torch.abs(b_triton - b_manual)).item()

print(f"Baseline max error: {error_baseline:.2e}")
print(f"Triton max error:   {error_triton:.2e}")

if error_baseline < 1e-4:
    print("✓ Baseline is correct (sequential execution)")
else:
    print("❌ Baseline has bugs!")

if error_triton < 1e-4:
    print("✓ Triton is correct")
else:
    print("❌ Triton has bugs!")
    print("\nLikely cause: Parallelizing elements within the same dependency chain")
    print("For example: b[8] and b[4] computed in parallel, but b[8] needs updated b[4]")

print("="*80)
