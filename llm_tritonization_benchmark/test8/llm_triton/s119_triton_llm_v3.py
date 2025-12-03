import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets >= 1) & (offsets < LEN_2D)
    
    aa_curr_idx = i_val * LEN_2D + offsets
    aa_prev_idx = (i_val - 1) * LEN_2D + (offsets - 1)
    bb_idx = i_val * LEN_2D + offsets
    
    aa_prev_vals = tl.load(aa_ptr + aa_prev_idx, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_idx, mask=mask)
    
    result = aa_prev_vals + bb_vals
    
    tl.store(aa_ptr + aa_curr_idx, result, mask=mask)

def s119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128
    
    for i_val in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s119_kernel[grid](aa, bb, i_val, LEN_2D, BLOCK_SIZE)