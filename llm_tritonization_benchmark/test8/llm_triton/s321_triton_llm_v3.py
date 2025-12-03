import torch
import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a sequential recurrence that cannot be parallelized
    # Process one element at a time from index 1 to n_elements-1
    
    for i in range(1, n_elements):
        # Load a[i-1], a[i], and b[i]
        prev_val = tl.load(a_ptr + (i - 1))
        curr_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        
        # Compute a[i] += a[i-1] * b[i]
        new_val = curr_val + prev_val * b_val
        
        # Store the result back to a[i]
        tl.store(a_ptr + i, new_val)

def s321_triton(a, b):
    n_elements = a.shape[0]
    
    # Use a single thread since this is a sequential recurrence
    BLOCK_SIZE = 1
    grid = (1,)
    
    s321_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )