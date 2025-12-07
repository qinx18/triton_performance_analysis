import torch
import triton
import triton.language as tl

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < LEN_2D * LEN_2D
    
    # Convert linear index to 2D coordinates
    i = indices // LEN_2D
    j = indices % LEN_2D
    
    # Calculate 2D array offsets
    aa_offsets = i * LEN_2D + j
    bb_offsets = i * LEN_2D + j
    cc_offsets = i * LEN_2D + j
    
    # Load values
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
    
    # Compute: aa[i][j] + bb[i][j] * cc[i][j]
    result = aa_vals + bb_vals * cc_vals
    
    # Store to flat array
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