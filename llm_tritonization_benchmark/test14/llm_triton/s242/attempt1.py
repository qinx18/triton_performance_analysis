import triton
import triton.language as tl
import torch

@triton.jit
def s242_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, s1, s2):
    # This kernel processes all elements sequentially due to RAW dependency
    # Each element depends on the previous element: a[i] = a[i-1] + ...
    
    # Sequential processing from i=1 to n_elements-1
    for i in range(1, n_elements):
        # Load a[i-1]
        prev_a = tl.load(a_ptr + i - 1)
        
        # Load b[i], c[i], d[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Compute a[i] = a[i-1] + s1 + s2 + b[i] + c[i] + d[i]
        result = prev_a + s1 + s2 + b_val + c_val + d_val
        
        # Store a[i]
        tl.store(a_ptr + i, result)

def s242_triton(a, b, c, d, s1, s2):
    n_elements = a.shape[0]
    
    # Launch kernel with single program
    grid = (1,)
    s242_kernel[grid](
        a, b, c, d,
        n_elements, s1, s2
    )
    
    return a