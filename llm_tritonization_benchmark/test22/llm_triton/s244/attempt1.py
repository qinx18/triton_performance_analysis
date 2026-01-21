import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Sequential processing for WAR dependency
    for i in range(N - 1):
        # S0: a[i] = b[i] + c[i] * d[i]
        # S1: b[i] = c[i] + b[i]
        # Load values for iteration i
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_next_val = tl.load(a_ptr + i + 1)
        
        # Execute S0
        a_val = b_val + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Execute S1
        b_new = c_val + b_val
        tl.store(b_ptr + i, b_new)
        
        # Execute S2 only for last iteration (i == N-2)
        if i == N - 2:
            a_next_new = b_new + a_next_val * d_val
            tl.store(a_ptr + i + 1, a_next_new)

def s244_triton(a, b, c, d):
    N = a.shape[0]
    
    # Launch single thread kernel due to sequential dependencies
    grid = (1,)
    s244_kernel[grid](a, b, c, d, N, BLOCK_SIZE=1)