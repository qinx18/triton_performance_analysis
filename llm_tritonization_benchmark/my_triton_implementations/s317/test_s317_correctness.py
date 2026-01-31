#!/usr/bin/env python3
"""
Correctness Test for s317
Compares Triton implementation against original TSVC C reference.
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    from c_reference.tsvc_all_reference import s317_c
    from test28.llm_triton.s317.attempt2 import s317_triton
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
    print(f"Correctness Testing: s317")
    print("Comparing Triton vs TSVC C reference")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            iterations = 1
            n = N

            pass

            pass

            c_tensors = {}
            tr_tensors = {}
            scalars = {"iterations": iterations, "n": n}

            c_kwargs = build_kwargs(s317_c, c_tensors, scalars)
            tr_kwargs = build_kwargs(s317_triton, tr_tensors, scalars)

            c_result = s317_c(**c_kwargs)
            triton_result = s317_triton(**tr_kwargs)

            # Pure scalar function - compare return values directly
            c_val = float(c_result) if c_result is not None else 0.0
            if isinstance(triton_result, (int, float)):
                tr_val = float(triton_result)
            elif isinstance(triton_result, torch.Tensor):
                tr_val = triton_result.item() if triton_result.numel() == 1 else float(triton_result)
            else:
                tr_val = float(triton_result) if triton_result is not None else 0.0
            max_error = abs(c_val - tr_val)
            is_scalar_comparison = True

            passed = max_error < 0.001 or (abs(c_val) > 1e-6 and max_error / abs(c_val) < 0.001)
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
