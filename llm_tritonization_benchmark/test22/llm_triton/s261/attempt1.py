import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependency
    # c[i] depends on c[i-1] from previous iteration
    # Only one thread processes all elements sequentially
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process all elements sequentially in a single thread
    for i in range(1, n):
        # Load values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_prev = tl.load(c_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # First computation: t = a[i] + b[i]
        t = a_val + b_val
        
        # Update a[i] = t + c[i-1]
        a_result = t + c_prev
        tl.store(a_ptr + i, a_result)
        
        # Second computation: t = c[i] * d[i]
        t = c_val * d_val
        
        # Update c[i] = t
        tl.store(c_ptr + i, t)

def s261_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single block since we need sequential processing
    grid = (1,)
    s261_kernel[grid](a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)