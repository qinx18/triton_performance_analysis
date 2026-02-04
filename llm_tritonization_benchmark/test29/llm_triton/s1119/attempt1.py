import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_start = pid * BLOCK_SIZE
    j_indices = j_start + j_offsets
    j_mask = j_indices < len_2d
    
    for i in range(1, len_2d):
        aa_prev_ptrs = aa_ptr + (i - 1) * len_2d + j_indices
        bb_curr_ptrs = bb_ptr + i * len_2d + j_indices
        aa_curr_ptrs = aa_ptr + i * len_2d + j_indices
        
        aa_prev = tl.load(aa_prev_ptrs, mask=j_mask, other=0.0)
        bb_curr = tl.load(bb_curr_ptrs, mask=j_mask, other=0.0)
        
        result = aa_prev + bb_curr
        
        tl.store(aa_curr_ptrs, result, mask=j_mask)

def s1119_triton(aa, bb, len_2d):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s1119_kernel[grid](
        aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )