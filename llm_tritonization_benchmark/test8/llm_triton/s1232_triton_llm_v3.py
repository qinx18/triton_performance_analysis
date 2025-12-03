import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get the current column (j)
    j = tl.program_id(0)
    
    if j >= LEN_2D:
        return
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process rows from j to LEN_2D with blocking
    for i_start in range(j, LEN_2D, BLOCK_SIZE):
        i_offsets = i_start + offsets
        i_mask = (i_offsets < LEN_2D) & (i_offsets >= j)
        
        # Calculate linear indices for [i][j] access
        linear_indices = i_offsets * LEN_2D + j
        
        # Load bb[i][j] and cc[i][j]
        bb_vals = tl.load(bb_ptr + linear_indices, mask=i_mask)
        cc_vals = tl.load(cc_ptr + linear_indices, mask=i_mask)
        
        # Compute aa[i][j] = bb[i][j] + cc[i][j]
        aa_vals = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + linear_indices, aa_vals, mask=i_mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (LEN_2D,)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D, BLOCK_SIZE
    )