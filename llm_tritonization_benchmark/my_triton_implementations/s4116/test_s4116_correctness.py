#!/usr/bin/env python3
"""
Correctness Test for s4116
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s4116_baseline import s4116_pytorch
    from test10.llm_triton.s4116.attempt1 import s4116_triton
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
    print(f"Correctness Testing: s4116")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            a = torch.randn(N + 10, device='cuda', dtype=torch.float32)
            aa = torch.randn(N + 10, N + 10, device='cuda', dtype=torch.float32)
            ip = torch.randint(0, N + 10, (N + 10,), device='cuda', dtype=torch.long)
            inc = 1
            iterations = 1
            j = 1

            a_pt = a.clone()
            aa_pt = aa.clone()
            ip_pt = ip.clone()

            a_tr = a.clone()
            aa_tr = aa.clone()
            ip_tr = ip.clone()

            pt_tensors = {"a": a_pt, "aa": aa_pt, "ip": ip_pt}
            tr_tensors = {"a": a_tr, "aa": aa_tr, "ip": ip_tr}
            scalars = {"inc": inc, "iterations": iterations, "j": j}

            pt_args = build_args(s4116_pytorch, pt_tensors, scalars)
            tr_args = build_args(s4116_triton, tr_tensors, scalars)

            pytorch_result = s4116_pytorch(*pt_args)
            s4116_triton(*tr_args)

            max_error = torch.max(torch.abs(a_pt - a_tr)).item()

            passed = max_error < 1e-3 or torch.allclose(a_pt, a_tr, rtol=1e-3, atol=1e-3)
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
