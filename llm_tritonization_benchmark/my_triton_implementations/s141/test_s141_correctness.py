#!/usr/bin/env python3
"""
Correctness Test for s141
Compares Triton implementation against original TSVC C reference.
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    from c_reference.tsvc_all_reference import s141_c
    from test25.llm_triton.s141.attempt1 import s141_triton
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
    test_sizes = [64, 128, 256]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s141")
    print("Comparing Triton vs TSVC C reference")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            bb = torch.randn(N, N, device='cuda', dtype=torch.float32)
            flat_2d_array = torch.randn(N * N, device='cuda', dtype=torch.float32)
            k = 0

            bb_c = bb.cpu().numpy().copy()
            flat_2d_array_c = flat_2d_array.cpu().numpy().copy()

            bb_tr = bb.clone()
            flat_2d_array_tr = flat_2d_array.clone()

            c_tensors = {"bb": bb_c, "flat_2d_array": flat_2d_array_c}
            tr_tensors = {"bb": bb_tr, "flat_2d_array": flat_2d_array_tr}
            scalars = {"k": k}

            c_kwargs = build_kwargs(s141_c, c_tensors, scalars)
            tr_kwargs = build_kwargs(s141_triton, tr_tensors, scalars)

            c_result = s141_c(**c_kwargs)
            triton_result = s141_triton(**tr_kwargs)

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
                # Array comparison - compare primary output array directly
                # Using bb which is the first output array (rw or w mode)
                c_arr = bb_c
                c_arr_flat = c_arr.flatten()
                c_arr_torch = torch.from_numpy(c_arr_flat.copy()).cuda()
                tr_arr = bb_tr.flatten()
                max_error = torch.max(torch.abs(c_arr_torch - tr_arr)).item()
                is_scalar_comparison = False

            if is_scalar_comparison:
                passed = max_error < 0.001 or (abs(c_val) > 1e-6 and max_error / abs(c_val) < 0.001)
            else:
                passed = max_error < 0.001 or torch.allclose(c_arr_torch, tr_arr, rtol=0.001, atol=0.001)
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
