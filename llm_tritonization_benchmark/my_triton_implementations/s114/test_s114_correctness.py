#!/usr/bin/env python3
"""
Correctness Test for s114
Tests: PyTorch baseline vs Triton LLM implementation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s114_baseline import s114_pytorch
    from llm_triton.s114_triton_llm import s114_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_correctness():
    """Test correctness across multiple sizes"""
    test_sizes = [100, 1000, 10000]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s114")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            # Initialize arrays according to TSVC initialise_arrays("s114")
            # aa: 1/(i+1) for each element (reciprocal of row index)
            # bb: 1/((i+1)^2) for each element (reciprocal squared of row index)
            # Vectorized initialization
            indices = torch.arange(1, N + 1, device='cuda', dtype=torch.float32)
            aa = (1.0 / indices).unsqueeze(1).expand(N, N).contiguous()
            bb = (1.0 / (indices * indices)).unsqueeze(1).expand(N, N).contiguous()

            # Run PyTorch baseline
            pytorch_result = s114_pytorch(aa.clone(), bb.clone())

            # Run Triton LLM
            triton_result = s114_triton(aa.clone(), bb.clone())

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
