#!/usr/bin/env python3
"""
Correctness Test for s4114
Tests: PyTorch baseline vs Triton LLM implementation (in-place comparison)
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s4114_baseline import s4114_pytorch
    from llm_triton.s4114_triton_llm_v3 import s4114_triton
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
    print(f"Correctness Testing: s4114")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            # Initialize base arrays
            a = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            b = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            c = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            d = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            ip = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            iterations = 1  # Scalar parameter (integer)
            n1 = 10  # Loop start offset

            # Create copies for PyTorch baseline
            a_pt = a.clone()
            b_pt = b.clone()
            c_pt = c.clone()
            d_pt = d.clone()
            ip_pt = ip.clone()

            # Create copies for Triton implementation
            a_tr = a.clone()
            b_tr = b.clone()
            c_tr = c.clone()
            d_tr = d.clone()
            ip_tr = ip.clone()

            # Available tensors and scalars for dynamic argument building
            pt_tensors = {"a": a_pt, "b": b_pt, "c": c_pt, "d": d_pt, "ip": ip_pt}
            tr_tensors = {"a": a_tr, "b": b_tr, "c": c_tr, "d": d_tr, "ip": ip_tr}
            scalars = {"iterations": iterations, "n1": n1}

            # Build argument lists based on actual function signatures
            pt_args = build_args(s4114_pytorch, pt_tensors, scalars)
            tr_args = build_args(s4114_triton, tr_tensors, scalars)

            # Run PyTorch baseline (may modify arrays in-place or return result)
            pytorch_result = s4114_pytorch(*pt_args)

            # Run Triton LLM (modifies arrays in-place)
            s4114_triton(*tr_args)

            # Compare output arrays directly (in-place modification)
            max_error = torch.max(torch.abs(a_pt - a_tr)).item()

            # Use relative tolerance for numerically unstable algorithms
            passed = max_error < 1e-3 or torch.allclose(a_pt, a_tr, rtol=1e-3, atol=1e-3)
            if passed:
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
