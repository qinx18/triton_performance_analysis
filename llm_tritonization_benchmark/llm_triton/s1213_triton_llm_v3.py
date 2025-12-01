import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Create temporary array for storing intermediate a values
    temp_a = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process sequentially to handle dependencies
    for block_start in range(1, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = (current_offsets >= 1) & (current_offsets < n_elements - 1)
        
        # Load required values
        b_prev = tl.load(b_ptr + current_offsets - 1, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        a_next = tl.load(a_ptr + current_offsets + 1, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = b[i-1] + c[i]
        new_a = b_prev + c_vals
        tl.store(a_ptr + current_offsets, new_a, mask=mask)
        
        # Compute b[i] = a[i+1] * d[i]
        new_b = a_next * d_vals
        tl.store(b_ptr + current_offsets, new_b, mask=mask)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s1213_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )