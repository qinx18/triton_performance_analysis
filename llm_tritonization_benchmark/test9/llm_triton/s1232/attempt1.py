import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the j dimension
    j = tl.program_id(0)
    
    if j >= LEN_2D:
        return
    
    # Process elements from i=j to i=LEN_2D-1 in blocks
    for i_start in range(j, LEN_2D, BLOCK_SIZE):
        i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
        
        # Mask for valid i indices (i < LEN_2D and i >= j)
        i_mask = (i_offsets < LEN_2D) & (i_offsets >= j)
        
        # Calculate linear indices for [i][j] access
        linear_indices = i_offsets * LEN_2D + j
        
        # Load bb[i][j] and cc[i][j]
        bb_vals = tl.load(bb_ptr + linear_indices, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + linear_indices, mask=i_mask, other=0.0)
        
        # Compute aa[i][j] = bb[i][j] + cc[i][j]
        aa_vals = bb_vals + cc_vals
        
        # Store aa[i][j]
        tl.store(aa_ptr + linear_indices, aa_vals, mask=i_mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.size(0)
    
    BLOCK_SIZE = 32
    
    # Launch kernel with one thread block per j value
    grid = (LEN_2D,)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa