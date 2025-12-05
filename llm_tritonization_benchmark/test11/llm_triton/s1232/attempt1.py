import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(LEN_2D):
        # Only process valid i indices where i >= j
        valid_mask = (i_idx < LEN_2D) & (i_idx >= j)
        
        # Calculate linear indices for 2D arrays
        linear_idx = i_idx * LEN_2D + j
        
        # Load data
        bb_vals = tl.load(bb_ptr + linear_idx, mask=valid_mask)
        cc_vals = tl.load(cc_ptr + linear_idx, mask=valid_mask)
        
        # Compute
        aa_vals = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + linear_idx, aa_vals, mask=valid_mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE
    )