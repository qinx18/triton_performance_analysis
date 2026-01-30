import triton
import triton.language as tl
import torch

@triton.jit
def s242_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s1, s2, n):
    # This kernel must run sequentially due to loop-carried dependency
    # Use single thread to process all elements sequentially
    
    for i in range(1, n):
        # Load a[i-1] from previous iteration
        prev_a = tl.load(a_ptr + (i - 1))
        
        # Load current elements
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Compute new value
        new_val = prev_a + s1 + s2 + b_val + c_val + d_val
        
        # Store result
        tl.store(a_ptr + i, new_val)

def s242_triton(a, b, c, d, s1, s2):
    n = a.shape[0]
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    s242_kernel[grid](a, b, c, d, s1, s2, n)
    
    return a