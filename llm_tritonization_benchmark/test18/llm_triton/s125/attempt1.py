import torch
import triton
import triton.language as tl

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, 
                len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    k_offsets = block_start + offsets
    
    mask = k_offsets < len_2d * len_2d
    
    # Convert linear index k to 2D indices (i, j)
    i_indices = k_offsets // len_2d
    j_indices = k_offsets % len_2d
    
    # Calculate 2D array offsets
    aa_offsets = i_indices * len_2d + j_indices
    bb_offsets = i_indices * len_2d + j_indices
    cc_offsets = i_indices * len_2d + j_indices
    
    # Load values from 2D arrays
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
    
    # Compute: aa[i][j] + bb[i][j] * cc[i][j]
    result = aa_vals + bb_vals * cc_vals
    
    # Store to flat array
    tl.store(flat_2d_array_ptr + k_offsets, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    len_2d = aa.shape[0]
    total_elements = len_2d * len_2d
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        aa, bb, cc, flat_2d_array,
        len_2d, BLOCK_SIZE
    )
    
    return flat_2d_array