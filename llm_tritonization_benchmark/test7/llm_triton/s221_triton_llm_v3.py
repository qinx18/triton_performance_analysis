import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This loop has a recurrence in b[i] = b[i-1] + a[i] + d[i]
    # Must process sequentially, cannot parallelize
    
    # Process one element at a time to handle the recurrence
    for i in range(1, n_elements):
        # Load current elements
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_val = tl.load(a_ptr + i)
        b_prev = tl.load(b_ptr + i - 1)
        
        # Compute a[i] += c[i] * d[i]
        new_a = a_val + c_val * d_val
        tl.store(a_ptr + i, new_a)
        
        # Compute b[i] = b[i-1] + a[i] + d[i]
        new_b = b_prev + new_a + d_val
        tl.store(b_ptr + i, new_b)

def s221_triton(a, b, c, d):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single block due to sequential nature
    
    s221_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )