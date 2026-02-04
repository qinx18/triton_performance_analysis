import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This is a sequential recurrence - use single thread
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Process sequentially from i=2 to N-1
    for i in range(2, N):
        # Load values
        a_i = tl.load(a_ptr + i)
        a_prev = tl.load(a_ptr + i - 1)
        a_prev2 = tl.load(a_ptr + i - 2)
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        
        # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_i + a_prev * b_i + a_prev2 * c_i
        
        # Store result
        tl.store(a_ptr + i, result)

def s322_triton(a, b, c):
    N = a.shape[0]
    
    if N <= 2:
        return a
    
    BLOCK_SIZE = 32
    grid = (1,)  # Single thread only due to sequential dependency
    
    s322_kernel[grid](
        a, b, c, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a