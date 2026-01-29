import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a serial recurrence that cannot be parallelized
    # We need to process elements sequentially
    for i in range(1, n_elements):
        # Load a[i-1], a[i], and b[i]
        prev_a = tl.load(a_ptr + (i - 1))
        curr_a = tl.load(a_ptr + i)
        curr_b = tl.load(b_ptr + i)
        
        # Compute a[i] += a[i-1] * b[i]
        result = curr_a + prev_a * curr_b
        
        # Store back to a[i]
        tl.store(a_ptr + i, result)

def s321_triton(a, b):
    n_elements = a.shape[0]
    
    # This recurrence must be processed sequentially due to dependencies
    # Launch a single thread block to handle the entire computation
    BLOCK_SIZE = 1
    grid = (1,)
    
    s321_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a