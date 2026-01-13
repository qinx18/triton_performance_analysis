import triton
import triton.language as tl
import torch

@triton.jit
def s242_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s1, s2, N):
    # Sequential processing - single thread handles all elements
    pid = tl.program_id(0)
    if pid > 0:  # Only first thread processes
        return
    
    # Process sequentially from i=1 to N-1
    for i in range(1, N):
        # Load previous value a[i-1]
        prev_val = tl.load(a_ptr + (i - 1))
        
        # Load current values
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Compute new value
        new_val = prev_val + s1 + s2 + b_val + c_val + d_val
        
        # Store result
        tl.store(a_ptr + i, new_val)

def s242_triton(a, b, c, d, s1, s2):
    N = a.shape[0]
    
    # Launch with single thread since this is strictly sequential
    grid = (1,)
    
    s242_kernel[grid](
        a, b, c, d,
        s1, s2, N
    )
    
    return a