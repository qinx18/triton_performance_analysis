import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n):
    # This kernel processes all iterations sequentially due to RAW dependency
    # Only one thread should execute this to maintain sequential order
    
    if tl.program_id(0) != 0:
        return
    
    # Process sequentially from i=1 to n-1
    for i in range(1, n):
        # Load values for current iteration
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_val = tl.load(a_ptr + i)
        b_prev = tl.load(b_ptr + i - 1)
        
        # Compute updates
        a_new = a_val + c_val * d_val
        b_new = b_prev + a_new + d_val
        
        # Store results
        tl.store(a_ptr + i, a_new)
        tl.store(b_ptr + i, b_new)

def s221_triton(a, b, c, d):
    n = a.shape[0]
    
    grid = (1,)  # Only one thread block needed for sequential processing
    s221_kernel[grid](a, b, c, d, n)
    
    return a, b