import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n):
    # This kernel processes one element at a time due to dependency in b[i] = b[i-1] + ...
    # We need to process sequentially
    for i in range(1, n):
        # Load values
        a_val = tl.load(a_ptr + i)
        b_prev = tl.load(b_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Compute a[i] += c[i] * d[i]
        new_a = a_val + c_val * d_val
        tl.store(a_ptr + i, new_a)
        
        # Compute b[i] = b[i-1] + a[i] + d[i]
        new_b = b_prev + new_a + d_val
        tl.store(b_ptr + i, new_b)

def s221_triton(a, b, c, d):
    n = a.shape[0]
    
    # Launch with single thread since we have sequential dependency
    grid = (1,)
    s221_kernel[grid](a, b, c, d, n)
    
    return a, b