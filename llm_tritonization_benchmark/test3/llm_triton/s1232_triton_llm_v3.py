import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for the j dimension
    j = tl.program_id(0)
    
    if j >= LEN_2D:
        return
    
    # Process elements in the i dimension where i >= j
    i_start = j
    num_elements = LEN_2D - j
    
    # Process in blocks along the i dimension
    for i_offset in range(0, num_elements, BLOCK_SIZE):
        i_indices = i_start + i_offset + tl.arange(0, BLOCK_SIZE)
        mask = i_indices < LEN_2D
        
        # Calculate linear indices for 2D arrays
        linear_indices = i_indices * LEN_2D + j
        
        # Load data with masking
        bb_vals = tl.load(bb_ptr + linear_indices, mask=mask)
        cc_vals = tl.load(cc_ptr + linear_indices, mask=mask)
        
        # Compute aa[i][j] = bb[i][j] + cc[i][j]
        aa_vals = bb_vals + cc_vals
        
        # Store result with masking
        tl.store(aa_ptr + linear_indices, aa_vals, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with one program per j value
    grid = (LEN_2D,)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return aa