import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(a_ptr, b_ptr, n):
    # This is a sequential recurrence that cannot be parallelized
    # We need to process elements one by one
    for i in range(1, n):
        # Load a[i-1], a[i], and b[i]
        a_prev = tl.load(a_ptr + (i - 1))
        a_curr = tl.load(a_ptr + i)
        b_curr = tl.load(b_ptr + i)
        
        # Compute a[i] += a[i-1] * b[i]
        result = a_curr + a_prev * b_curr
        
        # Store result back to a[i]
        tl.store(a_ptr + i, result)

def s321_triton(a, b):
    n = a.shape[0]
    
    # Launch kernel with single program instance since this is sequential
    s321_kernel[(1,)](a, b, n)
    
    return a