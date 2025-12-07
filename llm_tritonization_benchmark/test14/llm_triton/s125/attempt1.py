import torch
import triton
import triton.language as tl

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    k_offsets = block_start + offsets
    
    mask = k_offsets < LEN_2D * LEN_2D
    
    # Convert k to i,j coordinates
    i = k_offsets // LEN_2D
    j = k_offsets % LEN_2D
    
    # Calculate 2D array offset
    array_2d_offsets = i * LEN_2D + j
    
    # Load values from 2D arrays
    aa_vals = tl.load(aa_ptr + array_2d_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + array_2d_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + array_2d_offsets, mask=mask)
    
    # Compute result
    result = aa_vals + bb_vals * cc_vals
    
    # Store to flat array
    tl.store(flat_2d_array_ptr + k_offsets, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    LEN_2D = aa.shape[0]
    total_elements = LEN_2D * LEN_2D
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    
    s125_kernel[(grid_size,)](
        aa, bb, cc, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )