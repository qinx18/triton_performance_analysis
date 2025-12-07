import torch
import triton
import triton.language as tl

@triton.jit
def s125_kernel(
    aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr,
    LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    k_offsets = block_start + offsets
    
    # Mask for valid elements
    mask = k_offsets < (LEN_2D * LEN_2D)
    
    # Convert linear index k to 2D indices (i, j)
    i_indices = k_offsets // LEN_2D
    j_indices = k_offsets % LEN_2D
    
    # Calculate 2D array offsets
    array_2d_offsets = i_indices * LEN_2D + j_indices
    
    # Load data from 2D arrays
    aa_vals = tl.load(aa_ptr + array_2d_offsets, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + array_2d_offsets, mask=mask, other=0.0)
    cc_vals = tl.load(cc_ptr + array_2d_offsets, mask=mask, other=0.0)
    
    # Compute: aa[i][j] + bb[i][j] * cc[i][j]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result to flat array
    tl.store(flat_2d_array_ptr + k_offsets, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    LEN_2D = aa.shape[0]
    total_elements = LEN_2D * LEN_2D
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    
    # Launch kernel
    s125_kernel[(grid_size,)](
        aa, bb, cc, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )