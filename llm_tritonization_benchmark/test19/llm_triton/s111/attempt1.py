import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n):
    # This is a strictly sequential pattern - use single thread
    if tl.program_id(0) != 0:
        return
    
    # Process all iterations sequentially
    for i in range(1, n, 2):
        if i < n:
            # Load a[i-1] and b[i]
            a_prev = tl.load(a_ptr + (i - 1))
            b_val = tl.load(b_ptr + i)
            
            # Compute and store a[i] = a[i-1] + b[i]
            result = a_prev + b_val
            tl.store(a_ptr + i, result)

def s111_triton(a, b):
    n = a.shape[0]
    
    # Use single thread since this is strictly sequential
    grid = (1,)
    s111_kernel[grid](a, b, n)
    
    return a