#!/usr/bin/env python3
"""
Correctness Test for s2244
Compares Triton implementation against original TSVC C reference.
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    from c_reference.tsvc_all_reference import s2244_c
    from test23.llm_triton.s2244.attempt1 import s2244_triton
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
    print(f"Correctness Testing: s2244")
    print("Comparing Triton vs TSVC C reference")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            a = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            b = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            c = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            e = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            iterations = 1

            a_c = a.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()
            c_c = c.cpu().numpy().copy()
            e_c = e.cpu().numpy().copy()

            a_tr = a.clone()
            b_tr = b.clone()
            c_tr = c.clone()
            e_tr = e.clone()

            c_tensors = {"a": a_c, "b": b_c, "c": c_c, "e": e_c}
            tr_tensors = {"a": a_tr, "b": b_tr, "c": c_tr, "e": e_tr}
            scalars = {"iterations": iterations}

            c_args = build_args(s2244_c, c_tensors, scalars)
            tr_args = build_args(s2244_triton, tr_tensors, scalars)

            c_result = s2244_c(*c_args)
            triton_result = s2244_triton(*tr_args)

            # Convert C result back to torch for comparison
            # Use c_result if C function returns modified array, otherwise use in-place modified array
            c_arr = c_result if c_result is not None else a_c
            a_c_torch = torch.from_numpy(c_arr).cuda()
            max_error = torch.max(torch.abs(a_c_torch - a_tr)).item()

            passed = max_error < 1e-3 or torch.allclose(a_c_torch, a_tr, rtol=1e-3, atol=1e-3)
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
