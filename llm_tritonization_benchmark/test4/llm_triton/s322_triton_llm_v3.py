import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This is a second-order linear recurrence that must be computed sequentially
    # We cannot parallelize this computation due to dependencies on a[i-1] and a[i-2]
    
    # Process sequentially from index 2 to n-1
    for i in range(2, n):
        # Load current values
        a_i = tl.load(a_ptr + i)
        a_i_minus_1 = tl.load(a_ptr + i - 1)
        a_i_minus_2 = tl.load(a_ptr + i - 2)
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        
        # Compute recurrence: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_i + a_i_minus_1 * b_i + a_i_minus_2 * c_i
        
        # Store result
        tl.store(a_ptr + i, result)

def s322_triton(a, b, c):
    n = a.shape[0]
    
    # Launch with single program since this must be computed sequentially
    grid = (1,)
    s322_kernel[grid](
        a, b, c, n,
        BLOCK_SIZE=1024
    )
    
    return a