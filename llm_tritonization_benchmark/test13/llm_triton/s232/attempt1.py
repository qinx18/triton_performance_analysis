import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_idx < LEN_2D
    
    for i in range(1, LEN_2D):
        i_mask = j_idx >= i
        combined_mask = j_mask & i_mask
        
        prev_ptrs = aa_ptr + j_idx * LEN_2D + (i - 1)
        prev_vals = tl.load(prev_ptrs, mask=combined_mask, other=0.0)
        
        bb_ptrs = bb_ptr + j_idx * LEN_2D + i
        bb_vals = tl.load(bb_ptrs, mask=combined_mask, other=0.0)
        
        result = prev_vals * prev_vals + bb_vals
        
        aa_ptrs = aa_ptr + j_idx * LEN_2D + i
        tl.store(aa_ptrs, result, mask=combined_mask)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
    
    s232_kernel[grid](aa, bb, LEN_2D, BLOCK_SIZE)