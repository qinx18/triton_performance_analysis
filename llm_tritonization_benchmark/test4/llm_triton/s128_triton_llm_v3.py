import torch
import triton
import triton.language as tl

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Sequential processing due to loop-carried dependencies
    for i in range(n):
        k = 2 * i
        
        # a[i] = b[k] - d[i]
        b_k = tl.load(b_ptr + k)
        d_i = tl.load(d_ptr + i)
        a_i = b_k - d_i
        tl.store(a_ptr + i, a_i)
        
        # b[k] = a[i] + c[k]
        c_k = tl.load(c_ptr + k)
        b_k_new = a_i + c_k
        tl.store(b_ptr + k, b_k_new)

def s128_triton(a, b, c, d):
    n = a.shape[0] // 2
    
    # Launch single program instance for sequential execution
    grid = (1,)
    s128_kernel[grid](
        a, b, c, d, n,
        BLOCK_SIZE=1
    )