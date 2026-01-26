import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements):
    # This kernel must be sequential due to loop-carried dependencies
    # Use single thread to process all elements
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process all elements sequentially in a single thread
    for i in range(1, n_elements):
        # Load values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_prev = tl.load(c_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # First computation: t = a[i] + b[i]; a[i] = t + c[i-1]
        t = a_val + b_val
        new_a = t + c_prev
        tl.store(a_ptr + i, new_a)
        
        # Second computation: t = c[i] * d[i]; c[i] = t
        t = c_val * d_val
        tl.store(c_ptr + i, t)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Launch kernel with single thread group
    grid = (1,)
    s261_kernel[grid](a, b, c, d, n_elements)
    
    return a, c