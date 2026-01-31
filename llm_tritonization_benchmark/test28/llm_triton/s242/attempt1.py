import triton
import triton.language as tl
import torch

@triton.jit
def s242_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s1, s2, n_elements):
    # This kernel must run sequentially due to loop-carried dependency
    # Use single thread to process all elements
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Process sequentially from i=1 to n_elements-1
    for i in range(1, n_elements):
        # Load a[i-1]
        a_prev = tl.load(a_ptr + i - 1)
        
        # Load b[i], c[i], d[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Compute a[i] = a[i-1] + s1 + s2 + b[i] + c[i] + d[i]
        result = a_prev + s1 + s2 + b_val + c_val + d_val
        
        # Store a[i]
        tl.store(a_ptr + i, result)

def s242_triton(a, b, c, d, s1, s2):
    n_elements = a.shape[0]
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    
    s242_kernel[grid](
        a, b, c, d,
        s1, s2,
        n_elements
    )
    
    return a