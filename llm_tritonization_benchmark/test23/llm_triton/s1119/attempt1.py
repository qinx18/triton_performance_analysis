import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < n_cols
    
    for i in range(1, n_rows):
        # Load from aa[i-1][j]
        prev_row_offsets = (i - 1) * n_cols + j_idx
        aa_prev_vals = tl.load(aa_ptr + prev_row_offsets, mask=j_mask, other=0.0)
        
        # Load from bb[i][j]
        curr_row_offsets = i * n_cols + j_idx
        bb_vals = tl.load(bb_ptr + curr_row_offsets, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[i-1][j] + bb[i][j]
        result = aa_prev_vals + bb_vals
        
        # Store to aa[i][j]
        tl.store(aa_ptr + curr_row_offsets, result, mask=j_mask)

def s1119_triton(aa, bb):
    n_rows, n_cols = aa.shape
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_cols, BLOCK_SIZE),)
    s1119_kernel[grid](aa, bb, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return aa