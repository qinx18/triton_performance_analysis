import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n):
    # This kernel must run sequentially due to loop-carried dependency
    # Use a single thread to process all elements
    pid = tl.program_id(0)
    
    # Only thread 0 processes the entire loop sequentially
    if pid == 0:
        for i in range(1, n):
            # Load current values
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            a_val = tl.load(a_ptr + i)
            
            # Update a[i]
            new_a = a_val + c_val * d_val
            tl.store(a_ptr + i, new_a)
            
            # Load b[i-1] and update b[i]
            b_prev = tl.load(b_ptr + (i - 1))
            new_b = b_prev + new_a + d_val
            tl.store(b_ptr + i, new_b)

def s221_triton(a, b, c, d):
    N = a.shape[0]
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    s221_kernel[grid](a, b, c, d, N)
    
    return a, b