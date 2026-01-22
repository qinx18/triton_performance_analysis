import triton
import triton.language as tl
import torch

@triton.jit
def s242_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s1, s2, N, BLOCK_SIZE: tl.constexpr):
    # This kernel processes sequentially due to loop-carried dependency
    # Use only the first thread to avoid race conditions
    if tl.program_id(0) == 0:
        for i in range(1, N):
            # Load a[i-1]
            a_prev = tl.load(a_ptr + (i - 1))
            # Load b[i], c[i], d[i]
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            # Compute a[i] = a[i-1] + s1 + s2 + b[i] + c[i] + d[i]
            result = a_prev + s1 + s2 + b_val + c_val + d_val
            # Store a[i]
            tl.store(a_ptr + i, result)

def s242_triton(a, b, c, d, s1, s2):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single block since we need sequential execution
    grid = (1,)
    s242_kernel[grid](a, b, c, d, s1, s2, N, BLOCK_SIZE=BLOCK_SIZE)