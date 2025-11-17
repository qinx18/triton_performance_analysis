#!/usr/bin/env python3
"""
Correctness Test for s173
Tests: PyTorch baseline vs Triton LLM implementation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s173_baseline import s173_pytorch
    from llm_triton.s173_triton_correct import s173_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_correctness():
    """Test correctness across multiple sizes"""
    test_sizes = [100, 1000, 10000]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s173 (CORRECTED VERSION)")
    print("="*70)

    # Test with both k values to verify correct handling of dependencies
    test_k_values = [
        ("k=N//2 (TSVC)", None),  # k = LEN_1D/2, no dependencies
        ("k=5 (dependencies)", 5)  # k < half_len, has dependencies
    ]

    for k_desc, k_base in test_k_values:
        print(f"\n--- Testing with {k_desc} ---")
        for N in test_sizes:
            k = N // 2 if k_base is None else k_base
            print(f"Testing N={N:>6}, k={k:>4}...", end=" ")

            try:
                # Initialize arrays according to TSVC initialise_arrays("s173")
                # a: all 1.0
                # b: 1/(i+1)^2 for each i (reciprocal of index squared)
                a = torch.ones(N, device='cuda', dtype=torch.float32)
                b = torch.tensor([1.0 / ((i+1) * (i+1)) for i in range(N)],
                                device='cuda', dtype=torch.float32)

                # Run PyTorch baseline
                pytorch_result = s173_pytorch(a.clone(), b.clone(), k)

                # Run Triton corrected
                triton_result = s173_triton(a.clone(), b.clone(), k)

                # Compare results
                if isinstance(pytorch_result, tuple):
                    # Multiple outputs
                    max_error = max([torch.max(torch.abs(p - t)).item()
                                   for p, t in zip(pytorch_result, triton_result)])
                else:
                    # Single output
                    max_error = torch.max(torch.abs(pytorch_result - triton_result)).item()

                # Check if within tolerance
                if max_error < 1e-3:  # Relaxed tolerance for complex functions
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

    return all_passed

if __name__ == "__main__":
    success = test_correctness()
    sys.exit(0 if success else 1)
