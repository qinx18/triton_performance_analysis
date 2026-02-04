import triton
import triton.language as tl
import torch

@triton.jit
def s125_kernel(
    aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr,
    len_2d,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < len_2d * len_2d
    
    # Convert linear index to 2D coordinates
    i = indices // len_2d
    j = indices % len_2d
    
    # Calculate 2D array offset
    aa_offset = i * len_2d + j
    bb_offset = i * len_2d + j
    cc_offset = i * len_2d + j
    
    # Load values
    aa_vals = tl.load(aa_ptr + aa_offset, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_offset, mask=mask)
    cc_vals = tl.load(cc_ptr + cc_offset, mask=mask)
    
    # Compute result
    result = aa_vals + bb_vals * cc_vals
    
    # Store to flat array
    tl.store(flat_2d_array_ptr + indices, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array, len_2d):
    n_elements = len_2d * len_2d
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        aa, bb, cc, flat_2d_array,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )