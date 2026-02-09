#!/usr/bin/env python3
"""Correctness test for ludcmp (Polybench) - attempt 5"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results.llm_triton.ludcmp.attempt5 import ludcmp_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libludcmp.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(A_c, b_c, x_c, y_c, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_A = ctypes.c_float * (120 * 120)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_b = ctypes.c_float * (120)
    c_arr_b = CType_b.in_dll(lib, 'b')
    src_b = np.ascontiguousarray(b_c, dtype=np.float32)
    ctypes.memmove(c_arr_b, src_b.ctypes.data, src_b.nbytes)
    CType_x = ctypes.c_float * (120)
    c_arr_x = CType_x.in_dll(lib, 'x')
    src_x = np.ascontiguousarray(x_c, dtype=np.float32)
    ctypes.memmove(c_arr_x, src_x.ctypes.data, src_x.nbytes)
    CType_y = ctypes.c_float * (120)
    c_arr_y = CType_y.in_dll(lib, 'y')
    src_y = np.ascontiguousarray(y_c, dtype=np.float32)
    ctypes.memmove(c_arr_y, src_y.ctypes.data, src_y.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "ludcmp_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_A = ctypes.c_float * (120 * 120)
    c_arr_A = CType_A.in_dll(lib, 'A')
    A_c[:] = np.frombuffer(c_arr_A, dtype=np.float32).reshape(120, 120).copy()
    CType_x = ctypes.c_float * (120)
    c_arr_x = CType_x.in_dll(lib, 'x')
    x_c[:] = np.frombuffer(c_arr_x, dtype=np.float32).reshape(120).copy()
    CType_y = ctypes.c_float * (120)
    c_arr_y = CType_y.in_dll(lib, 'y')
    y_c[:] = np.frombuffer(c_arr_y, dtype=np.float32).reshape(120).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            A = torch.randn(120, 120, device='cuda', dtype=torch.float32)
            b = torch.randn(120, device='cuda', dtype=torch.float32)
            x = torch.randn(120, device='cuda', dtype=torch.float32)
            y = torch.randn(120, device='cuda', dtype=torch.float32)
            N = 120

            # Clone for C reference
            A_c = A.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()

            # Clone for Triton
            A_tr = A.clone()
            b_tr = b.clone()
            x_tr = x.clone()
            y_tr = y.clone()

            # Run C reference
            run_c_reference(A_c, b_c, x_c, y_c, N)

            # Run Triton
            ludcmp_triton(A_tr, b_tr, x_tr, y_tr, N)

            # Compare output arrays
            max_error = 0.0
            c_val = torch.from_numpy(A_c).float()
            tr_val = A_tr.cpu().float()
            err = torch.max(torch.abs(c_val - tr_val)).item()
            max_error = max(max_error, err)
            c_val = torch.from_numpy(x_c).float()
            tr_val = x_tr.cpu().float()
            err = torch.max(torch.abs(c_val - tr_val)).item()
            max_error = max(max_error, err)
            c_val = torch.from_numpy(y_c).float()
            tr_val = y_tr.cpu().float()
            err = torch.max(torch.abs(c_val - tr_val)).item()
            max_error = max(max_error, err)

            if max_error < 1e-3:
                print(f"  Test {test_idx + 1}: PASS (max_error={max_error:.6e})")
            else:
                print(f"  Test {test_idx + 1}: FAIL (max_error={max_error:.6e})")
                all_passed = False

        except Exception as e:
            print(f"  Test {test_idx + 1}: ERROR - {e}")
            all_passed = False

    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    return all_passed

if __name__ == "__main__":
    test_correctness()
