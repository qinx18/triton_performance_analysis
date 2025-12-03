#!/usr/bin/env python3
"""
Correctness Test for s2233
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s2233_baseline import s2233_pytorch
    from test9.llm_triton.s2233.attempt2 import s2233_triton
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
    print(f"Correctness Testing: s2233")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            aa = torch.randn(N + 10, N + 10, device='cuda', dtype=torch.float32)
            bb = torch.randn(N + 10, N + 10, device='cuda', dtype=torch.float32)
            cc = torch.randn(N + 10, N + 10, device='cuda', dtype=torch.float32)
            iterations = 1

            aa_pt = aa.clone()
            bb_pt = bb.clone()
            cc_pt = cc.clone()

            aa_tr = aa.clone()
            bb_tr = bb.clone()
            cc_tr = cc.clone()

            pt_tensors = {"aa": aa_pt, "bb": bb_pt, "cc": cc_pt}
            tr_tensors = {"aa": aa_tr, "bb": bb_tr, "cc": cc_tr}
            scalars = {"iterations": iterations}

            pt_args = build_args(s2233_pytorch, pt_tensors, scalars)
            tr_args = build_args(s2233_triton, tr_tensors, scalars)

            pytorch_result = s2233_pytorch(*pt_args)
            s2233_triton(*tr_args)

            max_error = torch.max(torch.abs(aa_pt - aa_tr)).item()

            passed = max_error < 1e-3 or torch.allclose(aa_pt, aa_tr, rtol=1e-3, atol=1e-3)
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
