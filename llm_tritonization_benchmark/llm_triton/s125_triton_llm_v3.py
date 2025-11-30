import triton
import triton.language as tl
import torch

@triton.jit
def s125_kernel(
    flat_2d_array_ptr,
    aa_ptr,
    bb_ptr,
    cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D * LEN_2D
    
    # Convert linear index to 2D coordinates
    i = offsets // LEN_2D
    j = offsets % LEN_2D
    
    # Calculate 2D array indices
    aa_indices = i * LEN_2D + j
    bb_indices = i * LEN_2D + j
    cc_indices = i * LEN_2D + j
    
    # Load data
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask)
    cc_vals = tl.load(cc_ptr + cc_indices, mask=mask)
    
    # Compute: flat_2d_array[k] = aa[i][j] + bb[i][j] * cc[i][j]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result
    tl.store(flat_2d_array_ptr + offsets, result, mask=mask)

def s125_triton(flat_2d_array, aa, bb, cc):
    LEN_2D = aa.shape[0]
    total_elements = LEN_2D * LEN_2D
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        flat_2d_array,
        aa,
        bb,
        cc,
        LEN_2D,
        BLOCK_SIZE,
    )
    
    return flat_2d_array