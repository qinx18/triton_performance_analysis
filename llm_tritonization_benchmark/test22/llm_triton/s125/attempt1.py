import triton
import triton.language as tl
import torch

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, 
                LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate start offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask to handle boundary conditions
    mask = indices < LEN_2D * LEN_2D
    
    # Load values from 2D arrays using flat indexing
    aa_vals = tl.load(aa_ptr + indices, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + indices, mask=mask, other=0.0)
    cc_vals = tl.load(cc_ptr + indices, mask=mask, other=0.0)
    
    # Compute: aa[i][j] + bb[i][j] * cc[i][j]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result
    tl.store(flat_2d_array_ptr + indices, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    LEN_2D = aa.shape[0]
    total_elements = LEN_2D * LEN_2D
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        aa, bb, cc, flat_2d_array,
        LEN_2D, BLOCK_SIZE
    )