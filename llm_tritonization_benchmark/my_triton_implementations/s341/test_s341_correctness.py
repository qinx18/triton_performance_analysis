#!/usr/bin/env python3
"""
Correctness Test for s341
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.s341_baseline import s341_pytorch
    from test11.llm_triton.s341.attempt1 import s341_triton
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
    print(f"Correctness Testing: s341")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            a = torch.randn(N, device='cuda', dtype=torch.float32)
            b = torch.randn(N, device='cuda', dtype=torch.float32)
            iterations = 1

            a_pt = a.clone()
            b_pt = b.clone()

            a_tr = a.clone()
            b_tr = b.clone()

            pt_tensors = {"a": a_pt, "b": b_pt}
            tr_tensors = {"a": a_tr, "b": b_tr}
            scalars = {"iterations": iterations}

            pt_args = build_args(s341_pytorch, pt_tensors, scalars)
            tr_args = build_args(s341_triton, tr_tensors, scalars)

            pytorch_result = s341_pytorch(*pt_args)
            s341_triton(*tr_args)

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
