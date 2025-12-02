import torch
import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a sequential recurrence - cannot be parallelized
    # Each element depends on the previous element
    # Process sequentially element by element
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(1, n_elements):
        # Load current values
        if i < n_elements:
            prev_a = tl.load(a_ptr + (i - 1))
            curr_b = tl.load(b_ptr + i)
            curr_a = tl.load(a_ptr + i)
            
            # Compute: a[i] += a[i-1] * b[i]
            new_a = curr_a + prev_a * curr_b
            
            # Store result
            tl.store(a_ptr + i, new_a)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 128
    
    grid = (1,)  # Single thread block since this is sequential
    
    s321_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a