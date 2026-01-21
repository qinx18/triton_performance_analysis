import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Process main loop iterations sequentially
    for i in range(n - 1):
        # Load scalars for current iteration
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i) 
        d_i = tl.load(d_ptr + i)
        a_i_plus_1 = tl.load(a_ptr + i + 1)
        
        # Execute statements in order
        # S0: a[i] = b[i] + c[i] * d[i]
        a_i_new = b_i + c_i * d_i
        tl.store(a_ptr + i, a_i_new)
        
        # S1: b[i] = c[i] + b[i]
        b_i_new = c_i + b_i
        tl.store(b_ptr + i, b_i_new)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i]
        a_i_plus_1_new = b_i_new + a_i_plus_1 * d_i
        tl.store(a_ptr + i + 1, a_i_plus_1_new)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    
    if n <= 1:
        return
    
    BLOCK_SIZE = 128
    grid = (1,)
    
    s244_kernel[grid](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )