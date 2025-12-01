import triton
import triton.language as tl
import torch

@triton.jit
def s242_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, s1, s2, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Sequential dependency requires processing one element at a time
    # Start from index 1 as per the C code
    for i in range(1, n_elements):
        # Load a[i-1]
        prev_val = tl.load(a_ptr + (i - 1))
        
        # Load b[i], c[i], d[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Compute: a[i] = a[i-1] + s1 + s2 + b[i] + c[i] + d[i]
        result = prev_val + s1 + s2 + b_val + c_val + d_val
        
        # Store result
        tl.store(a_ptr + i, result)

def s242_triton(a, b, c, d, s1, s2):
    n_elements = a.numel()
    
    # Launch single thread due to sequential dependency
    grid = (1,)
    
    s242_kernel[grid](
        a, b, c, d, s1, s2, n_elements,
        BLOCK_SIZE=256,
    )
    
    return a