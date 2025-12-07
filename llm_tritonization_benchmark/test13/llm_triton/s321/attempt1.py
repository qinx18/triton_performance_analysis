import torch
import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a sequential recurrence - each element depends on the previous
    # We need to process elements one by one
    for i in range(1, n_elements):
        # Load a[i-1], a[i], and b[i]
        a_prev = tl.load(a_ptr + (i - 1))
        a_curr = tl.load(a_ptr + i)
        b_curr = tl.load(b_ptr + i)
        
        # Compute a[i] += a[i-1] * b[i]
        result = a_curr + a_prev * b_curr
        
        # Store back to a[i]
        tl.store(a_ptr + i, result)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s321_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a