import triton
import triton.language as tl
import torch

@triton.jit
def s242_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s1, s2, n_elements):
    # Sequential computation - must process all elements in order
    # Start from index 1 as per original loop
    for i in range(1, n_elements):
        # Load a[i-1] (previous value)
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
    n_elements = a.shape[0]
    
    # Launch with single program since computation is strictly sequential
    grid = (1,)
    
    s242_kernel[grid](
        a, b, c, d,
        s1, s2,
        n_elements
    )
    
    return a