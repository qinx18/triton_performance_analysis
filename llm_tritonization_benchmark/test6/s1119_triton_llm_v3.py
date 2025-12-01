import torch
import triton
import triton.language as tl

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, i_val, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < n_cols
    
    # Calculate addresses for aa[i_val][j] and aa[i_val-1][j]
    curr_row_offset = i_val * n_cols + j_offsets
    prev_row_offset = (i_val - 1) * n_cols + j_offsets
    
    # Load aa[i_val-1][j] and bb[i_val][j]
    prev_vals = tl.load(aa_ptr + prev_row_offset, mask=mask)
    bb_vals = tl.load(bb_ptr + curr_row_offset, mask=mask)
    
    # Compute aa[i_val][j] = aa[i_val-1][j] + bb[i_val][j]
    result = prev_vals + bb_vals
    
    # Store result
    tl.store(aa_ptr + curr_row_offset, result, mask=mask)

def s1119_triton(aa, bb):
    n_rows, n_cols = aa.shape
    BLOCK_SIZE = 256
    
    # Sequential loop over i dimension
    for i_val in range(1, n_rows):
        grid = (triton.cdiv(n_cols, BLOCK_SIZE),)
        s1119_kernel[grid](aa, bb, i_val, n_cols, BLOCK_SIZE)