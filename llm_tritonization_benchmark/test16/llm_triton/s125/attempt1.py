import torch
import triton
import triton.language as tl

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < len_2d * len_2d
    
    # Convert linear index to 2D coordinates
    i = indices // len_2d
    j = indices % len_2d
    
    # Calculate 2D array offsets
    array_2d_offsets = i * len_2d + j
    
    # Load values from 2D arrays
    aa_vals = tl.load(aa_ptr + array_2d_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + array_2d_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + array_2d_offsets, mask=mask)
    
    # Compute: aa[i][j] + bb[i][j] * cc[i][j]
    result = aa_vals + bb_vals * cc_vals
    
    # Store to flat array
    tl.store(flat_2d_array_ptr + indices, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    len_2d = aa.shape[0]
    total_elements = len_2d * len_2d
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        aa, bb, cc, flat_2d_array,
        len_2d, BLOCK_SIZE
    )