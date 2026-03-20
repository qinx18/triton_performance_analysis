import torch
import triton
import triton.language as tl

@triton.jit
def spmv_kernel(
    cols_ptr, row_offsets_ptr, vals_ptr, x_ptr, y_ptr,
    NROWS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    
    if row_id < NROWS:
        # Get row boundaries
        row_start = tl.load(row_offsets_ptr + row_id)
        row_end = tl.load(row_offsets_ptr + row_id + 1)
        
        # Compute dot product for this row
        sum_val = 0.0
        for k in range(row_start, row_end):
            col_idx = tl.load(cols_ptr + k)
            val = tl.load(vals_ptr + k)
            x_val = tl.load(x_ptr + col_idx)
            sum_val += val * x_val
        
        # Store result
        tl.store(y_ptr + row_id, sum_val)

def spmv_triton(cols, row_offsets, vals, x, y, NNZ_PER_ROW, NROWS):
    grid = (NROWS,)
    
    spmv_kernel[grid](
        cols, row_offsets, vals, x, y,
        NROWS=NROWS, BLOCK_SIZE=1
    )