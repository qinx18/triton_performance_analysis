#!/usr/bin/env python3
"""
Correctness Test for s141
Tests: PyTorch baseline vs Triton LLM implementation (in-place comparison)
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s141_baseline import s141_pytorch
    from llm_triton.s141_triton_llm_v3 import s141_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def get_func_params(func):
    """Get the parameter names a function accepts"""
    sig = inspect.signature(func)
    return list(sig.parameters.keys())

def build_args(func, available_tensors, available_scalars):
    """Build argument list based on what the function actually accepts"""
    params = get_func_params(func)
    args = []
    for p in params:
        if p in available_tensors:
            args.append(available_tensors[p])
        elif p in available_scalars:
            args.append(available_scalars[p])
    return args

def test_correctness():
    """Test correctness across multiple sizes"""
    test_sizes = [100, 1000, 10000]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s141")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            # Initialize base arrays
            bb = torch.randn(N, N, device='cuda', dtype=torch.float32)
            flat_2d_array = torch.randn(N * N, device='cuda', dtype=torch.float32)
            iterations = 1  # Scalar parameter (integer)

            # Create copies for PyTorch baseline
            bb_pt = bb.clone()
            flat_2d_array_pt = flat_2d_array.clone()

            # Create copies for Triton implementation
            bb_tr = bb.clone()
            flat_2d_array_tr = flat_2d_array.clone()

            # Available tensors and scalars for dynamic argument building
            pt_tensors = {"bb": bb_pt, "flat_2d_array": flat_2d_array_pt}
            tr_tensors = {"bb": bb_tr, "flat_2d_array": flat_2d_array_tr}
            scalars = {"iterations": iterations}

            # Build argument lists based on actual function signatures
            pt_args = build_args(s141_pytorch, pt_tensors, scalars)
            tr_args = build_args(s141_triton, tr_tensors, scalars)

            # Run PyTorch baseline (may modify arrays in-place or return result)
            pytorch_result = s141_pytorch(*pt_args)

            # Run Triton LLM (modifies arrays in-place)
            s141_triton(*tr_args)

            # Compare output arrays directly (in-place modification)
            max_error = torch.max(torch.abs(flat_2d_array_pt - flat_2d_array_tr)).item()

            # Check if within tolerance
            if max_error < 1e-3:  # Relaxed tolerance for complex functions
                print(f"✓ PASS  (max_err={max_error:.2e})")
            else:
                print(f"✗ FAIL  (max_error={max_error:.2e})")
                all_passed = False

        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
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
