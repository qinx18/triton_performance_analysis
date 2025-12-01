import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential recurrence - must process element by element
    for i in range(1, n_elements):
        # Load current element of a and b
        if i < n_elements:
            a_prev = tl.load(a_ptr + (i - 1))
            a_curr = tl.load(a_ptr + i)
            b_curr = tl.load(b_ptr + i)
            
            # Compute recurrence: a[i] += a[i-1] * b[i]
            new_val = a_curr + a_prev * b_curr
            tl.store(a_ptr + i, new_val)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 128
    
    grid = lambda meta: (1,)  # Single thread block for sequential processing
    
    s321_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a