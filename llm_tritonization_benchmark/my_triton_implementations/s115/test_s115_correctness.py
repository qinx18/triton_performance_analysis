#!/usr/bin/env python3
"""
Correctness Test for s115
Compares Triton implementation against original TSVC C reference.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    from c_reference.tsvc_all_reference import s115_c
    from test23.llm_triton.s115.attempt10 import s115_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_correctness():
    test_sizes = [64, 128, 256]
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: s115")
    print("Comparing Triton vs TSVC C reference")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={N:>6}...", end=" ")

        try:
            a = torch.randn(N, device='cuda', dtype=torch.float32)
            aa = torch.randn(N, N, device='cuda', dtype=torch.float32)

            a_c = a.cpu().numpy().copy()
            aa_c = aa.cpu().numpy().copy()

            a_tr = a.clone()
            aa_tr = aa.clone()

            # Call C reference with explicit len_2d parameter
            # The C reference needs len_2d=N, not sqrt(len(a))
            c_result = s115_c(a_c, aa_c, len_2d=N)

            # Call Triton implementation
            s115_triton(a_tr, aa_tr)

            # Convert C result back to torch for comparison
            c_arr = c_result if c_result is not None else a_c
            a_c_torch = torch.from_numpy(c_arr).cuda()

            # s115 is back substitution which causes exponential value growth
            # Must use relative tolerance, not absolute tolerance
            # Values can grow to 1e+10 or higher, making absolute error meaningless
            torch.cuda.synchronize()

            abs_diff = torch.abs(a_c_torch - a_tr)
            max_abs_error = torch.max(abs_diff).item()
            max_val = max(torch.max(torch.abs(a_c_torch)).item(), 1e-10)
            max_rel_error = max_abs_error / max_val

            # Use more lenient tolerance for back substitution (accumulating FP errors)
            passed = torch.allclose(a_c_torch, a_tr, rtol=1e-3, atol=1e-5)
            if passed:
                print(f"PASS  (rel_err={max_rel_error:.2e}, abs_err={max_abs_error:.2e})")
            else:
                print(f"FAIL  (rel_err={max_rel_error:.2e}, abs_err={max_abs_error:.2e})")
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
