import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks sequentially to handle dependencies
    for block_start in range(1, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n_elements - 1)
        
        # Load required values for a[i] = b[i-1] + c[i]
        b_prev = tl.load(b_ptr + current_offsets - 1, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # Compute and store a[i]
        a_vals = b_prev + c_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # Load required values for b[i] = a[i+1] * d[i]
        a_next = tl.load(a_ptr + current_offsets + 1, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Compute and store b[i]
        b_vals = a_next * d_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch kernel with single thread block to handle dependencies
    grid = (1,)
    s1213_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)