import torch
import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load b values at current indices
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Calculate im1 indices (wraps around)
    im1_indices = tl.where(indices == 0, n_elements - 1, indices - 1)
    b_im1_vals = tl.load(b_ptr + im1_indices, mask=mask)
    
    # Calculate im2 indices (wraps around)
    im2_indices = tl.where(indices == 0, n_elements - 2, 
                          tl.where(indices == 1, n_elements - 1, indices - 2))
    b_im2_vals = tl.load(b_ptr + im2_indices, mask=mask)
    
    # Compute result
    result = (b_vals + b_im1_vals + b_im2_vals) * 0.333
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s292_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s292_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )