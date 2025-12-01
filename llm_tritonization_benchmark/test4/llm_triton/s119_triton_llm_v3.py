import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = pid * BLOCK_SIZE + offsets + 1
    j_mask = j_offsets < LEN_2D
    
    # Current position: aa[i_val][j]
    aa_current_idx = i_val * LEN_2D + j_offsets
    
    # Previous position: aa[i_val-1][j-1]
    aa_prev_idx = (i_val - 1) * LEN_2D + (j_offsets - 1)
    
    # bb position: bb[i_val][j]
    bb_idx = i_val * LEN_2D + j_offsets
    
    # Load data
    aa_prev = tl.load(aa_ptr + aa_prev_idx, mask=j_mask, other=0.0)
    bb_val = tl.load(bb_ptr + bb_idx, mask=j_mask, other=0.0)
    
    # Compute: aa[i][j] = aa[i-1][j-1] + bb[i][j]
    result = aa_prev + bb_val
    
    # Store result
    tl.store(aa_ptr + aa_current_idx, result, mask=j_mask)

def s119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential over i, parallel over j
    for i_val in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
        s119_kernel[grid](
            aa, bb, i_val, LEN_2D, BLOCK_SIZE
        )