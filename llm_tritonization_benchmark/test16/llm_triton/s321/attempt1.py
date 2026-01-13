import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a sequential recurrence that cannot be parallelized
    # We process one element at a time
    for i in range(1, n_elements):
        # Load current values
        a_i = tl.load(a_ptr + i)
        a_prev = tl.load(a_ptr + i - 1)
        b_i = tl.load(b_ptr + i)
        
        # Compute recurrence
        result = a_i + a_prev * b_i
        
        # Store result
        tl.store(a_ptr + i, result)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch single block since this is a sequential recurrence
    grid = (1,)
    
    s321_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )