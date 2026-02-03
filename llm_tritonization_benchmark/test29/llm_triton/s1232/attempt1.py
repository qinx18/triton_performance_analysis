import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(len_2d):
        # Only process elements where i >= j (upper triangular including diagonal)
        i_mask = (i_idx >= j) & (i_idx < len_2d)
        
        # Calculate linear indices for 2D arrays
        linear_idx = i_idx * len_2d + j
        
        # Load values
        bb_vals = tl.load(bb_ptr + linear_idx, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + linear_idx, mask=i_mask, other=0.0)
        
        # Compute
        aa_vals = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + linear_idx, aa_vals, mask=i_mask)

def s1232_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 128
    
    # Grid size based on i dimension
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )