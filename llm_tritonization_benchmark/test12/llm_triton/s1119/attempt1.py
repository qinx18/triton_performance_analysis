import torch
import triton
import triton.language as tl

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    for i in range(1, LEN_2D):
        prev_row_ptr = aa_ptr + (i - 1) * LEN_2D + j_idx
        curr_row_ptr = aa_ptr + i * LEN_2D + j_idx
        bb_ptr_curr = bb_ptr + i * LEN_2D + j_idx
        
        prev_vals = tl.load(prev_row_ptr, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr_curr, mask=j_mask, other=0.0)
        
        result = prev_vals + bb_vals
        
        tl.store(curr_row_ptr, result, mask=j_mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1119_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )