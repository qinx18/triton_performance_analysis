import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n):
    BLOCK_SIZE = 256
    
    # Process sequentially since j values have dependencies
    for block_start in range(0, n, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, n)
        
        for i in range(block_start, block_end):
            # Calculate k based on iteration: k = 2*i
            k = 2 * i
            
            # a[i] = b[k] - d[i]
            b_val = tl.load(b_ptr + k)
            d_val = tl.load(d_ptr + i)
            a_val = b_val - d_val
            tl.store(a_ptr + i, a_val)
            
            # b[k] = a[i] + c[k]
            c_val = tl.load(c_ptr + k)
            b_new = a_val + c_val
            tl.store(b_ptr + k, b_new)

def s128_triton(a, b, c, d):
    n = a.shape[0] // 2
    
    # Launch kernel with single program
    grid = (1,)
    s128_kernel[grid](a, b, c, d, n)
    
    return a, b