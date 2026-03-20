#!/usr/bin/env python3
"""Correctness test for spmv (Real-World) - attempt 2"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from realworld_results.llm_triton.spmv.attempt2 import spmv_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "realworld_libs" / "libspmv.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(cols_c, row_offsets_c, vals_c, x_c, y_c, NNZ_PER_ROW, NROWS):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_cols = ctypes.c_int * (110592)
    c_arr_cols = CType_cols.in_dll(lib, 'cols')
    src_cols = np.ascontiguousarray(cols_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_cols, src_cols.ctypes.data, src_cols.nbytes)
    CType_row_offsets = ctypes.c_int * (4097)
    c_arr_row_offsets = CType_row_offsets.in_dll(lib, 'row_offsets')
    src_row_offsets = np.ascontiguousarray(row_offsets_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_row_offsets, src_row_offsets.ctypes.data, src_row_offsets.nbytes)
    CType_vals = ctypes.c_float * (110592)
    c_arr_vals = CType_vals.in_dll(lib, 'vals')
    src_vals = np.ascontiguousarray(vals_c, dtype=np.float32)
    ctypes.memmove(c_arr_vals, src_vals.ctypes.data, src_vals.nbytes)
    CType_x = ctypes.c_float * (4096)
    c_arr_x = CType_x.in_dll(lib, 'x')
    src_x = np.ascontiguousarray(x_c, dtype=np.float32)
    ctypes.memmove(c_arr_x, src_x.ctypes.data, src_x.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "spmv_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_y = ctypes.c_float * (4096)
    c_arr_y = CType_y.in_dll(lib, 'y')
    y_c[:] = np.frombuffer(c_arr_y, dtype=np.float32).reshape(4096).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            NROWS = 4096
            NNZ_PER_ROW = 27
            NNZ = NROWS * NNZ_PER_ROW
            NCOLS = NROWS
            # Build valid CSR structure: each row has exactly NNZ_PER_ROW entries
            row_offsets = torch.arange(0, (NROWS + 1) * NNZ_PER_ROW, NNZ_PER_ROW, device='cuda', dtype=torch.int32)
            # Column indices: centered around diagonal, clamped to [0, NCOLS-1]
            cols_list = []
            half_bw = NNZ_PER_ROW // 2
            for r in range(NROWS):
                col_start = max(0, r - half_bw)
                row_cols = list(range(col_start, min(col_start + NNZ_PER_ROW, NCOLS)))
                while len(row_cols) < NNZ_PER_ROW:
                    row_cols.append(row_cols[-1])
                cols_list.extend(row_cols[:NNZ_PER_ROW])
            cols = torch.tensor(cols_list, device='cuda', dtype=torch.int32)
            vals = torch.randn(NNZ, device='cuda', dtype=torch.float32) * 0.1
            x = torch.randn(NCOLS, device='cuda', dtype=torch.float32)
            y = torch.zeros(NROWS, device='cuda', dtype=torch.float32)
            NNZ_PER_ROW = 27
            NROWS = 4096

            # Clone for C reference
            cols_c = cols.cpu().numpy().copy()
            row_offsets_c = row_offsets.cpu().numpy().copy()
            vals_c = vals.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()

            # Clone for Triton
            cols_tr = cols.clone()
            row_offsets_tr = row_offsets.clone()
            vals_tr = vals.clone()
            x_tr = x.clone()
            y_tr = y.clone()

            # Run C reference
            run_c_reference(cols_c, row_offsets_c, vals_c, x_c, y_c, NNZ_PER_ROW, NROWS)

            # Run Triton
            spmv_triton(cols_tr, row_offsets_tr, vals_tr, x_tr, y_tr, NNZ_PER_ROW, NROWS)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(y_c).float()
            tr_val = y_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)

            # Pass if absolute error < atol OR relative error < rtol
            passed = (max_error < 0.0001) or (max_rel_error < 0.0001)
            if passed:
                print(f"  Test {test_idx + 1}: PASS (abs={max_error:.6e} rel={max_rel_error:.6e})")
            else:
                print(f"  Test {test_idx + 1}: FAIL (abs={max_error:.6e} rel={max_rel_error:.6e})")
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
