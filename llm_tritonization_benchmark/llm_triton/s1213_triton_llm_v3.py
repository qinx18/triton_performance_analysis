import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Load b_readonly for the entire computation
    b_readonly = tl.load(b_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_elements, other=0.0)
    
    # Process sequentially due to dependencies
    for i in range(1, n_elements - 1):
        # Load required values
        b_prev = tl.load(b_ptr + i - 1)
        c_i = tl.load(c_ptr + i)
        d_i = tl.load(d_ptr + i)
        
        # First statement: a[i] = b[i-1] + c[i]
        a_i = b_prev + c_i
        tl.store(a_ptr + i, a_i)
        
        # Second statement: b[i] = a[i+1] * d[i]
        # Use original b value from readonly copy
        a_next = tl.load(a_ptr + i + 1) if i + 1 < n_elements - 1 else b_readonly[i + 1] * d_i
        if i + 1 < n_elements - 1:
            # a[i+1] will be computed, use original b[i] * d[i+1]
            b_i_plus_1 = tl.load(b_ptr + i)
            c_i_plus_1 = tl.load(c_ptr + i + 1)
            a_next = b_i_plus_1 + c_i_plus_1
        else:
            a_next = tl.load(a_ptr + i + 1)
        
        b_new = a_next * d_i
        tl.store(b_ptr + i, b_new)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Use single thread for sequential computation due to dependencies
    BLOCK_SIZE = 1024
    
    # Launch kernel with single block since computation must be sequential
    s1213_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )