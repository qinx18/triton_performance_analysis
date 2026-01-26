import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, N):
    # This kernel must process sequentially due to loop-carried dependency
    # Use single thread to maintain sequential order
    pid = tl.program_id(0)
    
    # Only process with first thread to ensure sequential execution
    if pid != 0:
        return
    
    # Sequential loop - cannot be parallelized due to a[i] = a[i-1] + b[i]
    for i in range(1, N, 2):
        # Load a[i-1] and b[i]
        a_prev = tl.load(a_ptr + (i - 1))
        b_val = tl.load(b_ptr + i)
        
        # Compute and store a[i] = a[i-1] + b[i]
        result = a_prev + b_val
        tl.store(a_ptr + i, result)

def s111_triton(a, b):
    N = a.shape[0]
    
    # Launch with single block, single thread to ensure sequential execution
    grid = (1,)
    
    s111_kernel[grid](
        a, b, N
    )
    
    return a