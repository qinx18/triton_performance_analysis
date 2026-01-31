import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_idx < len_2d
    
    for i in range(1, len_2d):
        i_mask = j_idx >= i
        valid_mask = j_mask & i_mask
        
        prev_aa = tl.load(aa_ptr + j_idx * len_2d + (i - 1), mask=valid_mask, other=0.0)
        bb_val = tl.load(bb_ptr + j_idx * len_2d + i, mask=valid_mask, other=0.0)
        
        result = prev_aa * prev_aa + bb_val
        
        tl.store(aa_ptr + j_idx * len_2d + i, result, mask=valid_mask)

def s232_triton(aa, bb, len_2d):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(len_2d - 1, BLOCK_SIZE),)
    
    s232_kernel[grid](aa, bb, len_2d, BLOCK_SIZE)