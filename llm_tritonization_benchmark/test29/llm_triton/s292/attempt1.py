import torch
import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute im1 and im2 indices for each thread
    # im1 starts at n_elements-1, then becomes i-1 for i > 0
    # im2 starts at n_elements-2, then becomes i-2 for i > 1
    im1_indices = tl.where(offsets == 0, n_elements - 1, offsets - 1)
    im2_indices = tl.where(offsets == 0, n_elements - 2,
                          tl.where(offsets == 1, n_elements - 1, offsets - 2))
    
    # Load b[im1] and b[im2] values
    im1_mask = (im1_indices >= 0) & (im1_indices < n_elements) & mask
    im2_mask = (im2_indices >= 0) & (im2_indices < n_elements) & mask
    
    b_im1 = tl.load(b_ptr + im1_indices, mask=im1_mask, other=0.0)
    b_im2 = tl.load(b_ptr + im2_indices, mask=im2_mask, other=0.0)
    
    # Compute a[i] = (b[i] + b[im1] + b[im2]) * 0.333
    result = (b_vals + b_im1 + b_im2) * 0.333
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s292_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s292_kernel[grid](a, b, n_elements, BLOCK_SIZE)