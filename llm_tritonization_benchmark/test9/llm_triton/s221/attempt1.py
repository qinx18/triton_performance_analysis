import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the sequential dependency by processing one element at a time
    # Starting from i=1 since i=0 is not processed in the original loop
    
    for i in range(1, n_elements):
        # Load single elements
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_val = tl.load(a_ptr + i)
        
        # First computation: a[i] += c[i] * d[i]
        new_a_val = a_val + c_val * d_val
        tl.store(a_ptr + i, new_a_val)
        
        # Second computation: b[i] = b[i-1] + a[i] + d[i]
        b_prev = tl.load(b_ptr + i - 1)
        new_b_val = b_prev + new_a_val + d_val
        tl.store(b_ptr + i, new_b_val)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 128
    
    # Launch kernel with single thread block since we need sequential processing
    grid = (1,)
    s221_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)