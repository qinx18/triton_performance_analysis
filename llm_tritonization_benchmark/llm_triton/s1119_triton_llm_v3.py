import torch
import triton
import triton.language as tl

@triton.jit
def s1119_kernel(
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    i_val,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Calculate 2D indices for current row (i_val) and previous row (i_val-1)
    current_row_offsets = i_val * LEN_2D + j_offsets
    prev_row_offsets = (i_val - 1) * LEN_2D + j_offsets
    
    # Load from read-only copy of aa (previous row) and bb (current row)
    aa_prev_vals = tl.load(aa_copy_ptr + prev_row_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + current_row_offsets, mask=mask)
    
    # Compute: aa[i][j] = aa[i-1][j] + bb[i][j]
    result = aa_prev_vals + bb_vals
    
    # Store to original aa array
    tl.store(aa_ptr + current_row_offsets, result, mask=mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Sequential loop over i dimension (i from 1 to LEN_2D-1)
    for i_val in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s1119_kernel[grid](
            aa,
            aa_copy,
            bb,
            i_val,
            LEN_2D,
            BLOCK_SIZE,
        )