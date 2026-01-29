import triton
import triton.language as tl
import torch

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n * n
    
    # Convert linear index to 2D indices
    i = idx // n
    j = idx % n
    
    # Calculate 2D array offsets
    aa_offsets = i * n + j
    bb_offsets = i * n + j
    cc_offsets = i * n + j
    
    # Load values from 2D arrays
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask, other=0.0)
    
    # Compute the result
    result = aa_vals + bb_vals * cc_vals
    
    # Store to flat array
    tl.store(flat_2d_array_ptr + idx, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    n = aa.shape[0]
    total_elements = n * n
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        aa, bb, cc, flat_2d_array,
        n, BLOCK_SIZE
    )