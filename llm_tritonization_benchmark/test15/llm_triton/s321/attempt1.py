import torch
import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a sequential recurrence, cannot be parallelized
    # Process elements sequentially one at a time
    for i in range(1, n_elements):
        # Load a[i-1], a[i], and b[i]
        prev_a = tl.load(a_ptr + (i - 1))
        curr_a = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        
        # Compute a[i] += a[i-1] * b[i]
        result = curr_a + prev_a * b_val
        
        # Store back to a[i]
        tl.store(a_ptr + i, result)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single program since this is sequential
    grid = (1,)
    
    s321_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )