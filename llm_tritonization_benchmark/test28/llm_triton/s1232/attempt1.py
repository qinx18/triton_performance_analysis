import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(len_2d):
        mask = (i_idx < len_2d) & (i_idx >= j)
        
        bb_ptrs = bb_ptr + i_idx * len_2d + j
        cc_ptrs = cc_ptr + i_idx * len_2d + j
        aa_ptrs = aa_ptr + i_idx * len_2d + j
        
        bb_vals = tl.load(bb_ptrs, mask=mask)
        cc_vals = tl.load(cc_ptrs, mask=mask)
        
        result = bb_vals + cc_vals
        
        tl.store(aa_ptrs, result, mask=mask)

def s1232_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )