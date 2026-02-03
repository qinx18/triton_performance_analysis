import torch
import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute im1 indices (previous iteration values)
    im1_offsets = tl.where(offsets == 0, n_elements - 1, offsets - 1)
    im1_mask = mask
    b_im1_vals = tl.load(b_ptr + im1_offsets, mask=im1_mask)
    
    # Compute im2 indices (two iterations back)
    im2_offsets = tl.where(offsets == 0, n_elements - 2, 
                  tl.where(offsets == 1, n_elements - 1, offsets - 2))
    im2_mask = mask
    b_im2_vals = tl.load(b_ptr + im2_offsets, mask=im2_mask)
    
    # Compute result
    result = (b_vals + b_im1_vals + b_im2_vals) * 0.333
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s292_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s292_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )