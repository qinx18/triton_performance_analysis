import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j in range(LEN_2D):
        i_idx = pid * BLOCK_SIZE + i_offsets
        
        # Triangular loop: i starts from j, not 0
        i_valid = (i_idx >= j) & (i_idx < LEN_2D)
        
        # Calculate linear indices for 2D arrays: [i][j] = i * LEN_2D + j
        linear_idx = i_idx * LEN_2D + j
        
        bb_ptrs = bb_ptr + linear_idx
        cc_ptrs = cc_ptr + linear_idx
        aa_ptrs = aa_ptr + linear_idx
        
        bb_vals = tl.load(bb_ptrs, mask=i_valid, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=i_valid, other=0.0)
        
        result = bb_vals + cc_vals
        
        tl.store(aa_ptrs, result, mask=i_valid)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )