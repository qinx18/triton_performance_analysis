import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n):
    # This kernel processes all elements sequentially due to RAW dependency
    # Only use one thread to maintain sequential order
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially
    for i in range(n - 1):
        # Load values for current iteration
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        e_val = tl.load(e_ptr + i)
        a_val = tl.load(a_ptr + i)
        
        # Compute new values
        a_new = b_val + c_val
        b_new = c_val * e_val
        d_new = a_val * e_val
        
        # Store results
        tl.store(a_ptr + i + 1, a_new)
        tl.store(b_ptr + i, b_new)
        tl.store(d_ptr + i, d_new)

def s3251_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Launch with only 1 block since we need sequential processing
    grid = (1,)
    
    s3251_kernel[grid](
        a, b, c, d, e, n
    )
    
    return a, b, c, d, e