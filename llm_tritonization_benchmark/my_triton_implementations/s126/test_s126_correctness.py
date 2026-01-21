#!/usr/bin/env python3
"""
Correctness Test for s126
Compares Triton implementation against original TSVC C reference.
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    from c_reference.tsvc_all_reference import s126_c
    from test20.llm_triton.s126.attempt10 import s126_triton
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
    print(f"Correctness Testing: s126")
    print("Comparing Triton vs TSVC C reference")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            bb = torch.randn(N + 10, N + 10, device='cuda', dtype=torch.float32)
            cc = torch.randn(N + 10, N + 10, device='cuda', dtype=torch.float32)
            flat_2d_array = torch.randn((N + 10) * (N + 10), device='cuda', dtype=torch.float32)
            iterations = 1

            bb_c = bb.cpu().numpy().copy()
            cc_c = cc.cpu().numpy().copy()
            flat_2d_array_c = flat_2d_array.cpu().numpy().copy()

            bb_tr = bb.clone()
            cc_tr = cc.clone()
            flat_2d_array_tr = flat_2d_array.clone()

            c_tensors = {"bb": bb_c, "cc": cc_c, "flat_2d_array": flat_2d_array_c}
            tr_tensors = {"bb": bb_tr, "cc": cc_tr, "flat_2d_array": flat_2d_array_tr}
            scalars = {"iterations": iterations}

            c_args = build_args(s126_c, c_tensors, scalars)
            tr_args = build_args(s126_triton, tr_tensors, scalars)

            c_result = s126_c(*c_args)
            triton_result = s126_triton(*tr_args)

            # Convert C result back to torch for comparison
            bb_c_torch = torch.from_numpy(bb_c).cuda()
            max_error = torch.max(torch.abs(bb_c_torch - bb_tr)).item()

            passed = max_error < 1e-3 or torch.allclose(bb_c_torch, bb_tr, rtol=1e-3, atol=1e-3)
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
