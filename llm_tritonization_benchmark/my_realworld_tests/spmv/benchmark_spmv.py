#!/usr/bin/env python3
"""Performance Benchmark for spmv (Real-World)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from realworld_results.llm_triton.spmv.attempt2 import spmv_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "realworld_libs" / "libspmv.so"

def run_c_reference(cols_c, row_offsets_c, vals_c, x_c, y_c, NNZ_PER_ROW, NROWS):
    lib = ctypes.CDLL(str(C_LIB_PATH))
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
    pass
    func = getattr(lib, "spmv_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_y = ctypes.c_float * (4096)
    c_arr_y = CType_y.in_dll(lib, 'y')
    y_c[:] = np.frombuffer(c_arr_y, dtype=np.float32).reshape(4096).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

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

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            cols_c = cols.cpu().numpy().copy()
            row_offsets_c = row_offsets.cpu().numpy().copy()
            vals_c = vals.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()
            run_c_reference(cols_c, row_offsets_c, vals_c, x_c, y_c, NNZ_PER_ROW, NROWS)
        start = time.perf_counter()
        for _ in range(num_iterations):
            cols_c = cols.cpu().numpy().copy()
            row_offsets_c = row_offsets.cpu().numpy().copy()
            vals_c = vals.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()
            run_c_reference(cols_c, row_offsets_c, vals_c, x_c, y_c, NNZ_PER_ROW, NROWS)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            cols_tr = cols.clone()
            row_offsets_tr = row_offsets.clone()
            vals_tr = vals.clone()
            x_tr = x.clone()
            y_tr = y.clone()
            spmv_triton(cols_tr, row_offsets_tr, vals_tr, x_tr, y_tr, NNZ_PER_ROW, NROWS)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            cols_tr = cols.clone()
            row_offsets_tr = row_offsets.clone()
            vals_tr = vals.clone()
            x_tr = x.clone()
            y_tr = y.clone()
            spmv_triton(cols_tr, row_offsets_tr, vals_tr, x_tr, y_tr, NNZ_PER_ROW, NROWS)
        torch.cuda.synchronize()
        tr_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"Triton error: {e}")

    # Report
    speedup = c_time / tr_time if c_time and tr_time and tr_time > 0 else None
    c_ms = c_time * 1000 if c_time else -1
    tr_ms = tr_time * 1000 if tr_time else -1
    sp = speedup if speedup else -1

    print(f"C ref:   {c_ms:8.3f} ms")
    print(f"Triton:  {tr_ms:8.3f} ms")
    if speedup:
        print(f"Speedup: {speedup:8.2f}x")
    else:
        print(f"Speedup: N/A")
    print(f"BENCHMARK_RESULT:{c_ms:.6f},{tr_ms:.6f},{sp:.6f}")

if __name__ == "__main__":
    benchmark()
