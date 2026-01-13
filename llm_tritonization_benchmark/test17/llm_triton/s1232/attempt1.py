import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(LEN_2D):
        valid_i = (i_idx >= j) & (i_idx < LEN_2D)
        
        if tl.sum(valid_i.to(tl.int32)) > 0:
            aa_ptrs = aa_ptr + i_idx * LEN_2D + j
            bb_ptrs = bb_ptr + i_idx * LEN_2D + j
            cc_ptrs = cc_ptr + i_idx * LEN_2D + j
            
            bb_vals = tl.load(bb_ptrs, mask=valid_i, other=0.0)
            cc_vals = tl.load(cc_ptrs, mask=valid_i, other=0.0)
            
            result = bb_vals + cc_vals
            
            tl.store(aa_ptrs, result, mask=valid_i)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )