import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a sequential recurrence - cannot be parallelized
    # Each element depends on the previous element
    # Process sequentially one element at a time
    
    for i in range(1, n_elements):
        # Load a[i-1], a[i], and b[i]
        a_prev = tl.load(a_ptr + (i - 1))
        a_curr = tl.load(a_ptr + i)
        b_curr = tl.load(b_ptr + i)
        
        # Compute a[i] += a[i-1] * b[i]
        result = a_curr + a_prev * b_curr
        
        # Store result back to a[i]
        tl.store(a_ptr + i, result)

def s321_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (1,)  # Single thread block since this is sequential
    
    s321_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )