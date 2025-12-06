import torch
import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program id and compute base offset
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load b values for current indices
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute im1 and im2 indices for each element
    im1_offsets = tl.where(offsets == 0, n_elements - 1, offsets - 1)
    im2_offsets = tl.where(offsets == 0, n_elements - 2,
                          tl.where(offsets == 1, n_elements - 1, offsets - 2))
    
    # Load b values for im1 and im2 indices
    b_im1_vals = tl.load(b_ptr + im1_offsets, mask=mask)
    b_im2_vals = tl.load(b_ptr + im2_offsets, mask=mask)
    
    # Compute result
    result = (b_vals + b_im1_vals + b_im2_vals) * 0.333
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s292_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s292_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a