#!/usr/bin/env python3
"""
Correctness Test for s132
Compares Triton implementation against original TSVC C reference.
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    from c_reference.tsvc_all_reference import s132_c
    from test24.llm_triton.s132.attempt1 import s132_triton
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
    print(f"Correctness Testing: s132")
    print("Comparing Triton vs TSVC C reference")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            aa = torch.randn(N + 10, N + 10, device='cuda', dtype=torch.float32)
            b = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            c = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            iterations = 1
            j = 1
            k = 0

            aa_c = aa.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()
            c_c = c.cpu().numpy().copy()

            aa_tr = aa.clone()
            b_tr = b.clone()
            c_tr = c.clone()

            c_tensors = {"aa": aa_c, "b": b_c, "c": c_c}
            tr_tensors = {"aa": aa_tr, "b": b_tr, "c": c_tr}
            scalars = {"iterations": iterations, "j": j, "k": k}

            c_kwargs = build_kwargs(s132_c, c_tensors, scalars)
            tr_kwargs = build_kwargs(s132_triton, tr_tensors, scalars)

            c_result = s132_c(**c_kwargs)
            triton_result = s132_triton(**tr_kwargs)

            # Convert C result back to torch for comparison
            # Use c_result if C function returns modified array, otherwise use in-place modified array
            c_arr = c_result if c_result is not None else aa_c
            aa_c_torch = torch.from_numpy(c_arr).cuda()
            max_error = torch.max(torch.abs(aa_c_torch - aa_tr)).item()

            passed = max_error < 1e-3 or torch.allclose(aa_c_torch, aa_tr, rtol=1e-3, atol=1e-3)
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
