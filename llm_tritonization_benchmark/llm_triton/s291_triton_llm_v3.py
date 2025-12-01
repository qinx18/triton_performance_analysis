import triton
import triton.language as tl
import torch

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Handle first element (i=0) separately
    if tl.program_id(0) == 0:
        # For i=0, im1 = LEN_1D-1
        b_0 = tl.load(b_ptr)
        b_im1 = tl.load(b_ptr + n_elements - 1)
        a_0 = (b_0 + b_im1) * 0.5
        tl.store(a_ptr, a_0)
    
    # Process remaining elements in blocks
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Load b[im1] where im1 = i-1 for i >= 1
        im1_offsets = current_offsets - 1
        b_im1_vals = tl.load(b_ptr + im1_offsets, mask=mask)
        
        # Calculate a[i] = (b[i] + b[im1]) * 0.5
        a_vals = (b_vals + b_im1_vals) * 0.5
        
        # Store results
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s291_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )