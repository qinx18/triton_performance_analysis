import torch
import triton
import triton.language as tl

@triton.jit
def spmv_kernel(
    cols_ptr, row_offsets_ptr, vals_ptr, x_ptr, y_ptr,
    NROWS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Each block processes multiple rows
    block_start = tl.program_id(0) * BLOCK_SIZE
    row_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < NROWS
    
    # Skip if no valid rows in this block
    if tl.sum(row_mask) > 0:
        for row_idx in range(BLOCK_SIZE):
            row = block_start + row_idx
            if row < NROWS:
                # Get row boundaries
                row_start = tl.load(row_offsets_ptr + row)
                row_end = tl.load(row_offsets_ptr + row + 1)
                
                # Compute dot product for this row
                sum_val = 0.0
                for k in range(row_start, row_end):
                    col_idx = tl.load(cols_ptr + k)
                    val = tl.load(vals_ptr + k)
                    x_val = tl.load(x_ptr + col_idx)
                    sum_val += val * x_val
                
                # Store result
                tl.store(y_ptr + row, sum_val)

def spmv_triton(cols, row_offsets, vals, x, y, NNZ_PER_ROW, NROWS):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(NROWS, BLOCK_SIZE),)
    
    spmv_kernel[grid](
        cols, row_offsets, vals, x, y,
        NROWS=NROWS, BLOCK_SIZE=BLOCK_SIZE
    )