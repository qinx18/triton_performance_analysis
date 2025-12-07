import torch
import triton
import triton.language as tl

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements):
    # This kernel must run with a single thread due to strict sequential dependency
    # Each iteration depends on the result of the previous iteration
    
    # Process all odd indices sequentially (i = 1, 3, 5, ...)
    i = 1
    while i < n_elements:
        # Load a[i-1] and b[i]
        a_prev = tl.load(a_ptr + (i - 1))
        b_val = tl.load(b_ptr + i)
        
        # Compute a[i] = a[i-1] + b[i]
        result = a_prev + b_val
        
        # Store result
        tl.store(a_ptr + i, result)
        
        # Move to next odd index
        i += 2

def s111_triton(a, b):
    n_elements = a.shape[0]
    
    # Launch kernel with single thread due to sequential dependency
    grid = (1,)
    s111_kernel[grid](a, b, n_elements)
    
    return a