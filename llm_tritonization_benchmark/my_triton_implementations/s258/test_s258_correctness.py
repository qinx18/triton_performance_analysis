#!/usr/bin/env python3
"""
Correctness Test for s258
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s258_baseline import s258_pytorch
    from test13.llm_triton.s258.attempt2 import s258_triton
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
    print(f"Correctness Testing: s258")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            a = torch.randn(N, device='cuda', dtype=torch.float32)
            aa = torch.randn(N, N, device='cuda', dtype=torch.float32)
            b = torch.randn(N, device='cuda', dtype=torch.float32)
            c = torch.randn(N, device='cuda', dtype=torch.float32)
            d = torch.randn(N, device='cuda', dtype=torch.float32)
            e = torch.randn(N, device='cuda', dtype=torch.float32)
            iterations = 1

            a_pt = a.clone()
            aa_pt = aa.clone()
            b_pt = b.clone()
            c_pt = c.clone()
            d_pt = d.clone()
            e_pt = e.clone()

            a_tr = a.clone()
            aa_tr = aa.clone()
            b_tr = b.clone()
            c_tr = c.clone()
            d_tr = d.clone()
            e_tr = e.clone()

            pt_tensors = {"a": a_pt, "aa": aa_pt, "b": b_pt, "c": c_pt, "d": d_pt, "e": e_pt}
            tr_tensors = {"a": a_tr, "aa": aa_tr, "b": b_tr, "c": c_tr, "d": d_tr, "e": e_tr}
            scalars = {"iterations": iterations}

            pt_args = build_args(s258_pytorch, pt_tensors, scalars)
            tr_args = build_args(s258_triton, tr_tensors, scalars)

            pytorch_result = s258_pytorch(*pt_args)
            s258_triton(*tr_args)

            max_error = max([torch.max(torch.abs(b_pt - b_tr)).item(), torch.max(torch.abs(e_pt - e_tr)).item()])

            passed = max_error < 1e-3 or torch.allclose(b_pt, b_tr, rtol=1e-3, atol=1e-3)
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
