import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements):
    # This kernel must run sequentially due to loop-carried dependency
    # c[i] depends on c[i-1] from previous iteration
    
    # Single thread processes all elements sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially from i=1 to n_elements-1
    for i in range(1, n_elements):
        # Load values for current iteration
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_prev = tl.load(c_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # First computation: t = a[i] + b[i]
        t = a_val + b_val
        
        # Update a[i] = t + c[i-1]
        a_new = t + c_prev
        tl.store(a_ptr + i, a_new)
        
        # Second computation: t = c[i] * d[i]
        t = c_val * d_val
        
        # Update c[i] = t
        tl.store(c_ptr + i, t)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Launch with only one block since we need sequential execution
    grid = (1,)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements
    )
    
    return a, c