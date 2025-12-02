import torch
import triton
import triton.language as tl

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, LEN_2D, i_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Calculate offsets for aa[i][j], aa[i-1][j], and bb[i][j]
    aa_curr_offsets = i_val * LEN_2D + j_offsets
    aa_prev_offsets = (i_val - 1) * LEN_2D + j_offsets
    bb_offsets = i_val * LEN_2D + j_offsets
    
    # Load aa[i-1][j] and bb[i][j]
    aa_prev = tl.load(aa_ptr + aa_prev_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    
    # Compute aa[i][j] = aa[i-1][j] + bb[i][j]
    result = aa_prev + bb_vals
    
    # Store result to aa[i][j]
    tl.store(aa_ptr + aa_curr_offsets, result, mask=mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i from 1 to LEN_2D-1
    for i_val in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s1119_kernel[grid](aa, bb, LEN_2D, i_val, BLOCK_SIZE)