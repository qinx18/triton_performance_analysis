import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependency
    # b[i] depends on b[i-1] from previous iteration
    
    # Use single thread to process sequentially
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Process all elements sequentially in a single thread
    for i in range(1, n - 1):
        # Load values for current iteration
        b_prev = tl.load(b_ptr + (i - 1))
        c_curr = tl.load(c_ptr + i)
        a_next = tl.load(a_ptr + (i + 1))
        d_curr = tl.load(d_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i]
        a_val = b_prev + c_curr
        tl.store(a_ptr + i, a_val)
        
        # Compute b[i] = a[i+1] * d[i]
        b_val = a_next * d_curr
        tl.store(b_ptr + i, b_val)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    
    # Single thread grid since we need sequential execution
    grid = (1,)
    BLOCK_SIZE = 256
    
    s1213_kernel[grid](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )