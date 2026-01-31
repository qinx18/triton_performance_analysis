#!/usr/bin/env python3
"""
Correctness Test for s151
Compares Triton implementation against original TSVC C reference.
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    from c_reference.tsvc_all_reference import s151_c
    from test28.llm_triton.s151.attempt1 import s151_triton
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
    print(f"Correctness Testing: s151")
    print("Comparing Triton vs TSVC C reference")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            a = torch.randn(N, device='cuda', dtype=torch.float32)
            b = torch.randn(N, device='cuda', dtype=torch.float32)
            iterations = 1

            a_c = a.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()

            a_tr = a.clone()
            b_tr = b.clone()

            c_tensors = {"a": a_c, "b": b_c}
            tr_tensors = {"a": a_tr, "b": b_tr}
            scalars = {"iterations": iterations}

            c_kwargs = build_kwargs(s151_c, c_tensors, scalars)
            tr_kwargs = build_kwargs(s151_triton, tr_tensors, scalars)

            c_result = s151_c(**c_kwargs)
            triton_result = s151_triton(**tr_kwargs)

            # Collect post-execution arrays for checksum
            c_tensors_after = {"a": a_c, "b": b_c}
            tr_tensors_after = {"a": a_tr, "b": b_tr}

            # Runtime detection: compare scalars if C returns scalar, otherwise use checksum
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
                # C wrapper modifies 1D arrays in-place via ctypes pointers,
                # so c_tensors_after already has correct values for 1D arrays.
                # However, 2D arrays (aa, bb, cc) are flattened copies in the C wrapper,
                # so their modifications are NOT reflected in c_tensors_after.
                # Update c_tensors_after with any 2D arrays from the return value.
                _checksum_2d = [name for name, is_2d in [('a', False)] if is_2d]
                if c_result is not None and _checksum_2d:
                    _returns = (c_result,) if isinstance(c_result, np.ndarray) else (c_result if isinstance(c_result, tuple) else ())
                    _ret_2d = [r for r in _returns if isinstance(r, np.ndarray) and r.ndim == 2]
                    for _name, _arr in zip(_checksum_2d, _ret_2d):
                        c_tensors_after[_name] = _arr

                # Checksum-based comparison (matches TSVC_2 calc_checksum)
                c_checksum = float(np.sum(c_tensors_after['a']))
                tr_checksum = float(torch.sum(tr_tensors_after['a']).item())
                # Handle inf/nan: if both are same inf, treat as match
                import math
                if math.isinf(c_checksum) and math.isinf(tr_checksum) and (c_checksum > 0) == (tr_checksum > 0):
                    max_error = 0.0
                elif math.isnan(c_checksum) or math.isnan(tr_checksum):
                    max_error = float('inf')
                else:
                    max_error = abs(c_checksum - tr_checksum)
                    # Use relative tolerance for large checksums
                    if abs(c_checksum) > 1e-6:
                        max_error = max_error / abs(c_checksum)
                is_scalar_comparison = False

            if is_scalar_comparison:
                passed = max_error < 0.001 or (abs(c_val) > 1e-6 and max_error / abs(c_val) < 0.001)
            else:
                passed = max_error < 0.001
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
