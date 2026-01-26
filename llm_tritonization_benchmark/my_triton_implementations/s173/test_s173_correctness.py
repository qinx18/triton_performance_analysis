#!/usr/bin/env python3
"""
Correctness Test for s173
Compares Triton implementation against original TSVC C reference.
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    from c_reference.tsvc_all_reference import s173_c
    from test25.llm_triton.s173.attempt1 import s173_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def get_func_params(func):
    sig = inspect.signature(func)
    return list(sig.parameters.keys())

def build_kwargs(func, available_tensors, available_scalars):
    params = get_func_params(func)
    kwargs = {}
    for p in params:
        if p in available_tensors:
            kwargs[p] = available_tensors[p]
        elif p in available_scalars:
            kwargs[p] = available_scalars[p]
    return kwargs

def test_correctness():
    test_sizes = [100, 1000, 10000]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s173")
    print("Comparing Triton vs TSVC C reference")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            a = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            b = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            iterations = 1
            k = 0

            a_c = a.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()

            a_tr = a.clone()
            b_tr = b.clone()

            c_tensors = {"a": a_c, "b": b_c}
            tr_tensors = {"a": a_tr, "b": b_tr}
            scalars = {"iterations": iterations, "k": k}

            c_kwargs = build_kwargs(s173_c, c_tensors, scalars)
            tr_kwargs = build_kwargs(s173_triton, tr_tensors, scalars)

            c_result = s173_c(**c_kwargs)
            triton_result = s173_triton(**tr_kwargs)

            # Runtime detection: compare scalars if C returns scalar, otherwise compare arrays
            if isinstance(c_result, (int, float)):
                # Scalar return - compare values directly
                c_val = float(c_result)
                if isinstance(triton_result, (int, float)):
                    tr_val = float(triton_result)
                elif isinstance(triton_result, torch.Tensor):
                    tr_val = triton_result.item() if triton_result.numel() == 1 else float(triton_result)
                else:
                    tr_val = float(triton_result) if triton_result is not None else float('inf')
                max_error = abs(c_val - tr_val)
                is_scalar_comparison = True
            else:
                # Array comparison - C function modifies arrays in-place or returns array
                c_arr = c_result if isinstance(c_result, np.ndarray) else a_c
                a_c_torch = torch.from_numpy(c_arr).cuda()
                max_error = torch.max(torch.abs(a_c_torch - a_tr)).item()
                is_scalar_comparison = False

            if is_scalar_comparison:
                passed = max_error < 0.001 or (abs(c_val) > 1e-6 and max_error / abs(c_val) < 0.001)
            else:
                passed = max_error < 0.001 or torch.allclose(a_c_torch, a_tr, rtol=0.001, atol=0.001)
            if passed:
                print(f"PASS  (max_err={max_error:.2e})")
            else:
                print(f"FAIL  (max_error={max_error:.2e})")
                all_passed = False

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("="*70)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("="*70)

    return all_passed

if __name__ == "__main__":
    success = test_correctness()
    sys.exit(0 if success else 1)
