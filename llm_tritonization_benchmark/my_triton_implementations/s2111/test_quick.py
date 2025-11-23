#!/usr/bin/env python3
"""
Quick correctness test for s2111 with smaller sizes
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

from baselines.s2111_baseline import s2111_pytorch
from llm_triton.s2111_triton_diagonal import s2111_triton

test_sizes = [10, 50, 100, 500]
all_passed = True

print("="*70)
print(f"Quick Correctness Testing: s2111")
print("="*70)

for N in test_sizes:
    print(f"Testing N={N:>6}...", end=" ", flush=True)

    try:
        # Initialize arrays
        aa = torch.randn(N, N, device='cuda', dtype=torch.float32)

        # Run PyTorch baseline
        pytorch_result = s2111_pytorch(aa.clone())

        # Run Triton
        triton_result = s2111_triton(aa.clone())

        # Compare results
        max_error = torch.max(torch.abs(pytorch_result - triton_result)).item()

        # Check if within tolerance
        if max_error < 1e-3:
            print(f"✓ PASS  (max_err={max_error:.2e})")
        else:
            print(f"✗ FAIL  (max_error={max_error:.2e})")
            all_passed = False

    except Exception as e:
        print(f"✗ ERROR: {e}")
        all_passed = False

print("="*70)
if all_passed:
    print("✅ All tests PASSED!")
else:
    print("❌ Some tests FAILED!")
print("="*70)

sys.exit(0 if all_passed else 1)
