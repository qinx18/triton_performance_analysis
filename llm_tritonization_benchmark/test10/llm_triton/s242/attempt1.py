import torch
import triton
import triton.language as tl

@triton.jit
def s242_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, s1, s2, BLOCK_SIZE: tl.constexpr):
    # Sequential dependency - process one element at a time
    for i in range(1, n_elements):
        # Load previous value of a[i-1]
        prev_a = tl.load(a_ptr + (i - 1))
        
        # Load current values
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Compute new value
        new_val = prev_a + s1 + s2 + b_val + c_val + d_val
        
        # Store result
        tl.store(a_ptr + i, new_val)

def s242_triton(a, b, c, d, s1, s2):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single program
    s242_kernel[(1,)](
        a, b, c, d, n_elements, s1, s2, BLOCK_SIZE
    )
    
    return a