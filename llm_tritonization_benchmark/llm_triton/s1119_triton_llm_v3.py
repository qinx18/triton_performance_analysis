import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, i_val, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Calculate memory offsets for aa[i][j] and aa[i-1][j]
    curr_row_offset = i_val * n_cols
    prev_row_offset = (i_val - 1) * n_cols
    
    # Load aa[i-1][j] and bb[i][j]
    aa_prev = tl.load(aa_ptr + prev_row_offset + offsets, mask=mask)
    bb_curr = tl.load(bb_ptr + curr_row_offset + offsets, mask=mask)
    
    # Compute aa[i][j] = aa[i-1][j] + bb[i][j]
    result = aa_prev + bb_curr
    
    # Store result to aa[i][j]
    tl.store(aa_ptr + curr_row_offset + offsets, result, mask=mask)

def s1119_triton(aa, bb):
    n_rows, n_cols = aa.shape
    BLOCK_SIZE = 256
    
    # Sequential loop over i from 1 to n_rows-1
    for i in range(1, n_rows):
        grid = (triton.cdiv(n_cols, BLOCK_SIZE),)
        s1119_kernel[grid](aa, bb, i, n_cols, BLOCK_SIZE)