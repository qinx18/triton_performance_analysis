import triton
import triton.language as tl
import torch

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b[i] values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute im1 indices: (i-1) for i > 0, (n_elements-1) for i = 0
        im1_offsets = tl.where(current_offsets == 0, n_elements - 1, current_offsets - 1)
        
        # Load b[im1] values
        b_im1_vals = tl.load(b_ptr + im1_offsets, mask=mask)
        
        # Compute a[i] = (b[i] + b[im1]) * 0.5
        result = (b_vals + b_im1_vals) * 0.5
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s291_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )