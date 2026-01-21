#!/usr/bin/env python3
"""
Correctness Test for s3113
Compares Triton implementation against original TSVC C reference.
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    from c_reference.tsvc_all_reference import s3113_c
    from test23.llm_triton.s3113.attempt10 import s3113_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def get_func_params(func):
    sig = inspect.signature(func)
    return list(sig.parameters.keys())

def build_args(func, available_tensors, available_scalars):
    params = get_func_params(func)
    args = []
    for p in params:
        if p in available_tensors:
            args.append(available_tensors[p])
        elif p in available_scalars:
            args.append(available_scalars[p])
    return args

def test_correctness():
    test_sizes = [100, 1000, 10000]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s3113")
    print("Comparing Triton vs TSVC C reference")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            a = torch.randn(N, device='cuda', dtype=torch.float32)
            iterations = 1

            a_c = a.cpu().numpy().copy()

            a_tr = a.clone()

            c_tensors = {"a": a_c}
            tr_tensors = {"a": a_tr}
            scalars = {"iterations": iterations}

            c_args = build_args(s3113_c, c_tensors, scalars)
            tr_args = build_args(s3113_triton, tr_tensors, scalars)

            c_result = s3113_c(*c_args)
            triton_result = s3113_triton(*tr_args)

            # Pure reduction: compare return values
            # If C returns None (void function), use numpy sum as reference
            if c_result is None:
                # C function is void - use numpy sum as baseline reference
                c_val = float(np.sum(a_c))
            elif isinstance(c_result, (int, float)):
                c_val = c_result
            elif isinstance(c_result, np.ndarray):
                c_val = c_result.item() if c_result.size == 1 else c_result.sum()
            else:
                c_val = float(c_result)

            if triton_result is None:
                tr_val = float('inf')  # Triton should return something
            elif isinstance(triton_result, (int, float)):
                tr_val = triton_result
            elif isinstance(triton_result, torch.Tensor):
                tr_val = triton_result.item() if triton_result.numel() == 1 else triton_result.sum().item()
            else:
                tr_val = float(triton_result)

            max_error = abs(c_val - tr_val)

            passed = max_error < 1e-3 or (abs(c_val) > 1e-6 and max_error / abs(c_val) < 1e-3)
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
