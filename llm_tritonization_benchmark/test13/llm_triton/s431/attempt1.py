import triton
import triton.language as tl
import torch

@triton.jit
def s431_kernel(a_ptr, b_ptr, n_elements, k, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create masks
    mask = current_offsets < n_elements
    read_mask = (current_offsets + k) < n_elements
    combined_mask = mask & read_mask
    
    # Load data with masks
    a_vals = tl.load(a_ptr + current_offsets + k, mask=combined_mask, other=0.0)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s431_triton(a, b, k):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s431_kernel[grid](
        a, b, n_elements, k,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a