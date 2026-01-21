import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially
    if tl.program_id(0) != 0:
        return
    
    # Process sequentially from i=1 to i<n-1
    for i in range(1, n - 1):
        # First statement: a[i] = b[i-1] + c[i]
        b_prev = tl.load(b_ptr + (i - 1))
        c_val = tl.load(c_ptr + i)
        a_val = b_prev + c_val
        tl.store(a_ptr + i, a_val)
        
        # Second statement: b[i] = a[i+1] * d[i]
        a_next = tl.load(a_ptr + (i + 1))
        d_val = tl.load(d_ptr + i)
        b_val = a_next * d_val
        tl.store(b_ptr + i, b_val)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 1
    
    # Launch with single thread
    grid = (1,)
    
    s1213_kernel[grid](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )