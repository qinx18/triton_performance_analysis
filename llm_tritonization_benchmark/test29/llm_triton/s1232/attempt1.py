import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(len_2d):
        # Compute mask for triangular access pattern (i >= j)
        mask = (i_idx >= j) & (i_idx < len_2d)
        
        # Calculate linear indices for 2D arrays
        linear_idx = i_idx * len_2d + j
        
        # Load data
        bb_vals = tl.load(bb_ptr + linear_idx, mask=mask)
        cc_vals = tl.load(cc_ptr + linear_idx, mask=mask)
        
        # Compute
        result = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + linear_idx, result, mask=mask)

def s1232_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 256
    i_size = len_2d
    grid = (triton.cdiv(i_size, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc, len_2d, BLOCK_SIZE
    )