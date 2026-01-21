import triton
import triton.language as tl
import torch

@triton.jit
def s242_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s1, s2, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must run with a single thread since the computation is strictly sequential
    pid = tl.program_id(0)
    
    # Only process with the first thread
    if pid != 0:
        return
    
    # Sequential loop - each iteration depends on the previous
    for i in range(1, n):
        # Load a[i-1]
        prev_val = tl.load(a_ptr + i - 1)
        
        # Load b[i], c[i], d[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Compute a[i] = a[i-1] + s1 + s2 + b[i] + c[i] + d[i]
        result = prev_val + s1 + s2 + b_val + c_val + d_val
        
        # Store a[i]
        tl.store(a_ptr + i, result)

def s242_triton(a, b, c, d, s1, s2):
    n = a.shape[0]
    
    # Launch with only one thread block since computation is strictly sequential
    grid = (1,)
    BLOCK_SIZE = 256
    
    s242_kernel[grid](
        a, b, c, d, 
        s1, s2, n, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a