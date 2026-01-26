import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, N):
    # This kernel must be strictly sequential due to RAW dependencies
    # Use a single thread to process all elements sequentially
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process sequentially from index 2 to N-1
    for i in range(2, N):
        # Load current values
        a_curr = tl.load(a_ptr + i)
        b_curr = tl.load(b_ptr + i)
        c_curr = tl.load(c_ptr + i)
        
        # Load previous values (these may have been updated in previous iterations)
        a_prev1 = tl.load(a_ptr + i - 1)
        a_prev2 = tl.load(a_ptr + i - 2)
        
        # Compute new value: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        new_val = a_curr + a_prev1 * b_curr + a_prev2 * c_curr
        
        # Store result
        tl.store(a_ptr + i, new_val)

def s322_triton(a, b, c):
    N = a.shape[0]
    
    # Launch with only 1 thread since we must process sequentially
    grid = (1,)
    
    s322_kernel[grid](
        a, b, c, N
    )
    
    return a