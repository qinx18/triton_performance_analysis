import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(b_ptr, a_ptr, n):
    # Define offsets once at kernel start
    offsets = tl.arange(0, 1)
    
    # Sequential computation due to dependencies
    im1 = n - 1
    im2 = n - 2
    
    for i in range(n):
        # Load b[i], b[im1], b[im2]
        i_offset = i + offsets
        im1_offset = im1 + offsets
        im2_offset = im2 + offsets
        
        i_mask = i_offset < n
        im1_mask = im1_offset < n
        im2_mask = im2_offset < n
        
        b_i = tl.load(b_ptr + i_offset, mask=i_mask)
        b_im1 = tl.load(b_ptr + im1_offset, mask=im1_mask)
        b_im2 = tl.load(b_ptr + im2_offset, mask=im2_mask)
        
        # Compute result
        result = (b_i + b_im1 + b_im2) * 0.333
        
        # Store result
        tl.store(a_ptr + i_offset, result, mask=i_mask)
        
        # Update indices
        im2 = im1
        im1 = i

def s292_triton(a, b):
    n = b.shape[0]
    
    # Launch kernel with single thread
    grid = (1,)
    s292_kernel[grid](b, a, n)
    
    return a