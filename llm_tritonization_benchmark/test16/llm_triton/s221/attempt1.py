import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n):
    # This kernel must run sequentially due to loop-carried dependency
    # Use single thread to process all elements
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one thread processes everything
        # Process sequentially from i=1 to n-1
        for i in range(1, n):
            # Load values for current iteration
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            a_val = tl.load(a_ptr + i)
            
            # Update a[i]
            new_a_val = a_val + c_val * d_val
            tl.store(a_ptr + i, new_a_val)
            
            # Load b[i-1] and compute b[i]
            b_prev = tl.load(b_ptr + (i - 1))
            new_b_val = b_prev + new_a_val + d_val
            tl.store(b_ptr + i, new_b_val)

def s221_triton(a, b, c, d):
    n = a.shape[0]
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    
    s221_kernel[grid](
        a, b, c, d, n
    )
    
    return a, b