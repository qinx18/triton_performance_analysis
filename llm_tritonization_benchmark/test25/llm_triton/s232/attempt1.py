import triton
import triton.language as tl
import torch

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_idx < len_2d
    
    for i in range(1, len_2d):
        valid_mask = j_mask & (j_idx >= i)
        
        if tl.sum(valid_mask.to(tl.int32)) > 0:
            prev_offsets = j_idx * len_2d + (i - 1)
            curr_offsets = j_idx * len_2d + i
            
            prev_vals = tl.load(aa_ptr + prev_offsets, mask=valid_mask, other=0.0)
            bb_vals = tl.load(bb_ptr + curr_offsets, mask=valid_mask, other=0.0)
            
            result = prev_vals * prev_vals + bb_vals
            
            tl.store(aa_ptr + curr_offsets, result, mask=valid_mask)

def s232_triton(aa, bb, len_2d):
    BLOCK_SIZE = 64
    j_size = len_2d - 1
    grid = (triton.cdiv(j_size, BLOCK_SIZE),)
    
    s232_kernel[grid](aa, bb, len_2d, BLOCK_SIZE)