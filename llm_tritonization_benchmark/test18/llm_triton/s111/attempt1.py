import torch
import triton
import triton.language as tl

@triton.jit
def s111_kernel(a_ptr, b_ptr, n):
    # This kernel must be strictly sequential due to loop-carried dependency
    # Use a single thread to process all iterations sequentially
    pid = tl.program_id(0)
    
    if pid == 0:  # Only first thread does the work
        # Process sequentially: i = 1, 3, 5, ... < n
        i = 1
        while i < n:
            # Load a[i-1] and b[i]
            a_prev = tl.load(a_ptr + (i - 1))
            b_val = tl.load(b_ptr + i)
            
            # Compute and store a[i]
            result = a_prev + b_val
            tl.store(a_ptr + i, result)
            
            i += 2

def s111_triton(a, b):
    n = a.shape[0]
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    s111_kernel[grid](a, b, n)
    
    return a