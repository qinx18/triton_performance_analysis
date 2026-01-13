import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n):
    # This kernel must be sequential due to loop-carried dependency
    # Use single thread to process all elements
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially from 1 to n-1
    for i in range(1, n):
        # Load values for current iteration
        b_prev = tl.load(b_ptr + (i - 1))
        c_i = tl.load(c_ptr + i)
        d_i = tl.load(d_ptr + i)
        e_i = tl.load(e_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_val = b_prev + c_i * d_i
        tl.store(a_ptr + i, a_val)
        
        # Compute b[i] = a[i] + c[i] * e[i]
        b_val = a_val + c_i * e_i
        tl.store(b_ptr + i, b_val)

def s323_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    s323_kernel[grid](
        a, b, c, d, e, n
    )
    
    return a, b