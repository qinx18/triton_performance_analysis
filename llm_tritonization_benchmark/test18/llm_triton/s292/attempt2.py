import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process each element sequentially due to dependencies
    for i in range(n_elements):
        # Compute im1 and im2 indices based on current position
        im1_idx = n_elements - 1 if i == 0 else i - 1
        im2_idx = n_elements - 2 if i == 0 else (n_elements - 1 if i == 1 else i - 2)
        
        # Load required values
        b_i = tl.load(b_ptr + i)
        b_im1 = tl.load(b_ptr + im1_idx)
        b_im2 = tl.load(b_ptr + im2_idx)
        
        # Compute result
        result = (b_i + b_im1 + b_im2) * 0.333
        
        # Store result
        tl.store(a_ptr + i, result)

def s292_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single program since we need sequential processing
    s292_kernel[(1,)](a, b, n_elements, BLOCK_SIZE)
    
    return a