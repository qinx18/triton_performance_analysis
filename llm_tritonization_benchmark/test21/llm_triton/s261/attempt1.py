import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependency
    # c[i] depends on c[i-1] from previous iteration
    
    # Use single thread to process all elements sequentially
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Sequential processing from i=1 to n-1
    for i in range(1, n):
        # Load values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_prev = tl.load(c_ptr + (i - 1))
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # First computation: t = a[i] + b[i]
        # Then: a[i] = t + c[i-1]
        t = a_val + b_val
        new_a = t + c_prev
        
        # Second computation: t = c[i] * d[i]
        # Then: c[i] = t
        t = c_val * d_val
        new_c = t
        
        # Store results
        tl.store(a_ptr + i, new_a)
        tl.store(c_ptr + i, new_c)

def s261_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use single block since processing must be sequential
    grid = (1,)
    
    s261_kernel[grid](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )