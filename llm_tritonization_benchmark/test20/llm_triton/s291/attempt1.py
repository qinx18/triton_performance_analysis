import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values for current block
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # For each element, we need b[im1] where im1 follows the pattern:
    # i=0: im1 = n_elements-1
    # i>0: im1 = i-1
    im1_offsets = tl.where(offsets == 0, n_elements - 1, offsets - 1)
    
    # Load b[im1] values
    b_im1_vals = tl.load(b_ptr + im1_offsets, mask=mask)
    
    # Compute a[i] = (b[i] + b[im1]) * 0.5
    result = (b_vals + b_im1_vals) * 0.5
    
    # Store results
    tl.store(a_ptr + offsets, result, mask=mask)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s291_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )