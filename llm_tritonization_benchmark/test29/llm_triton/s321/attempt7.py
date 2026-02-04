import torch
import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a sequential recurrence - must be computed serially
    for i in range(1, n_elements):
        # Load a[i-1], a[i], and b[i]
        prev_val = tl.load(a_ptr + (i-1))
        curr_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        
        # Compute a[i] += a[i-1] * b[i]
        new_val = curr_val + prev_val * b_val
        
        # Store result
        tl.store(a_ptr + i, new_val)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single thread since this is sequential
    s321_kernel[(1,)](
        a, b,
        n_elements,
        BLOCK_SIZE
    )
    
    return a