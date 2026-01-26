import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential recurrence - must process element by element
    for i in range(1, n_elements):
        # Load current values
        if i < BLOCK_SIZE:
            mask = offsets == i
            a_curr = tl.load(a_ptr + i)
            a_prev = tl.load(a_ptr + i - 1)
            b_curr = tl.load(b_ptr + i)
            
            # Compute: a[i] += a[i-1] * b[i]
            result = a_curr + a_prev * b_curr
            tl.store(a_ptr + i, result)
        else:
            # For elements beyond BLOCK_SIZE, process individually
            a_curr = tl.load(a_ptr + i)
            a_prev = tl.load(a_ptr + i - 1)
            b_curr = tl.load(b_ptr + i)
            
            result = a_curr + a_prev * b_curr
            tl.store(a_ptr + i, result)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch single instance since this is a sequential recurrence
    grid = (1,)
    s321_kernel[grid](a, b, n_elements, BLOCK_SIZE)