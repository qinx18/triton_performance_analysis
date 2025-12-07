import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load b values for current indices
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Calculate im1 indices (wrap around)
    im1_indices = tl.where(indices == 0, n_elements - 1, indices - 1)
    
    # Load b values for im1 indices
    b_im1_vals = tl.load(b_ptr + im1_indices, mask=mask)
    
    # Compute result
    result = (b_vals + b_im1_vals) * 0.5
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s291_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s291_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )