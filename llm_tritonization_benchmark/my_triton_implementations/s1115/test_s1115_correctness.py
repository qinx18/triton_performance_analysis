#!/usr/bin/env python3
"""
Correctness Test for s1115
Compares Triton implementation against original TSVC C reference.
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    from c_reference.tsvc_all_reference import s1115_c
    from test20.llm_triton.s1115.attempt10 import s1115_triton
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
    test_sizes = [64, 128, 256]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s1115")
    print("Comparing Triton vs TSVC C reference")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            aa = torch.randn(N, N, device='cuda', dtype=torch.float32)
            bb = torch.randn(N, N, device='cuda', dtype=torch.float32)
            cc = torch.randn(N, N, device='cuda', dtype=torch.float32)
            iterations = 1

            aa_c = aa.cpu().numpy().copy()
            bb_c = bb.cpu().numpy().copy()
            cc_c = cc.cpu().numpy().copy()

            aa_tr = aa.clone()
            bb_tr = bb.clone()
            cc_tr = cc.clone()

            c_tensors = {"aa": aa_c, "bb": bb_c, "cc": cc_c}
            tr_tensors = {"aa": aa_tr, "bb": bb_tr, "cc": cc_tr}
            scalars = {"iterations": iterations}

            c_args = build_args(s1115_c, c_tensors, scalars)
            tr_args = build_args(s1115_triton, tr_tensors, scalars)

            c_result = s1115_c(*c_args)
            triton_result = s1115_triton(*tr_args)

            # Convert C result back to torch for comparison
            aa_c_torch = torch.from_numpy(aa_c).cuda()
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
