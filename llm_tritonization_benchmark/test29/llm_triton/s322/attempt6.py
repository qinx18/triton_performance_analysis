import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Sequential processing from i=2 to N-1
    for i in range(2, N):
        # Load scalar values
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
    
    # Use single thread since this is strictly sequential
    grid = (1,)
    BLOCK_SIZE = 1
    
    s322_kernel[grid](
        a, b, c, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a