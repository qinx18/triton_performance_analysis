import torch
import triton
import triton.language as tl

@triton.jit
def spmv_kernel(
    cols_ptr,
    row_offsets_ptr,
    vals_ptr,
    x_ptr,
    y_ptr,
    NROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_idx < NROWS
    
    if tl.sum(mask) > 0:
        valid_rows = tl.sum(mask)
        for i in range(BLOCK_SIZE):
            if i < valid_rows:
                row = tl.program_id(0) * BLOCK_SIZE + i
                if row < NROWS:
                    row_start = tl.load(row_offsets_ptr + row)
                    row_end = tl.load(row_offsets_ptr + row + 1)
                    
                    sum_val = 0.0
                    for j in range(row_start, row_end):
                        col_idx = tl.load(cols_ptr + j)
                        val = tl.load(vals_ptr + j)
                        x_val = tl.load(x_ptr + col_idx)
                        sum_val += val * x_val
                    
                    tl.store(y_ptr + row, sum_val)

def spmv_triton(cols, row_offsets, vals, x, y, NNZ_PER_ROW, NROWS):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(NROWS, BLOCK_SIZE),)
    
    spmv_kernel[grid](
        cols,
        row_offsets,
        vals,
        x,
        y,
        NROWS,
        BLOCK_SIZE,
    )