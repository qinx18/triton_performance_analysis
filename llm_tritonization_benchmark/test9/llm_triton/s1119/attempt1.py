import torch
import triton
import triton.language as tl

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for i in range(1, len_2d):
        aa_curr_ptr = aa_ptr + i * len_2d + j_offsets
        aa_prev_ptr = aa_ptr + (i - 1) * len_2d + j_offsets
        bb_curr_ptr = bb_ptr + i * len_2d + j_offsets
        
        aa_prev_vals = tl.load(aa_prev_ptr, mask=j_mask, other=0.0)
        bb_curr_vals = tl.load(bb_curr_ptr, mask=j_mask, other=0.0)
        
        result = aa_prev_vals + bb_curr_vals
        
        tl.store(aa_curr_ptr, result, mask=j_mask)

def s1119_triton(aa, bb):
    len_2d = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    grid = (1,)
    
    s1119_kernel[grid](
        aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )