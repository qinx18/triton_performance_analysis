import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(a_ptr, b_ptr, n):
    # Sequential processing required due to recurrence dependency
    for i in range(1, n):
        # Load current values
        a_curr = tl.load(a_ptr + i)
        a_prev = tl.load(a_ptr + i - 1)
        b_curr = tl.load(b_ptr + i)
        
        # Compute recurrence
        result = a_curr + a_prev * b_curr
        
        # Store result
        tl.store(a_ptr + i, result)

def s321_triton(a, b):
    n = a.shape[0]
    
    # Launch kernel with single thread since this is sequential
    grid = (1,)
    s321_kernel[grid](a, b, n)
    
    return a