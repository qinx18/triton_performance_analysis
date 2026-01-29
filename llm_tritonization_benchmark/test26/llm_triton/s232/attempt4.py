import triton
import triton.language as tl
import torch

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_idx < LEN_2D
    
    for i in range(1, LEN_2D):
        i_mask = j_idx >= i
        valid_mask = j_mask & i_mask
        
        aa_prev_offsets = j_idx * LEN_2D + (i - 1)
        aa_prev = tl.load(aa_ptr + aa_prev_offsets, mask=valid_mask, other=0.0)
        
        bb_offsets = j_idx * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=valid_mask, other=0.0)
        
        aa_new = aa_prev * aa_prev + bb_vals
        
        aa_offsets = j_idx * LEN_2D + i
        tl.store(aa_ptr + aa_offsets, aa_new, mask=valid_mask)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
    
    s232_kernel[grid](aa, bb, LEN_2D, BLOCK_SIZE)
    
    return aa