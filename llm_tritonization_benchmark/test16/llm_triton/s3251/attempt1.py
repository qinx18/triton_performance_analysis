import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, N):
    # This kernel must process sequentially due to RAW dependency
    # Use single thread processing
    pid = tl.program_id(0)
    
    # Only process with first program
    if pid != 0:
        return
    
    # Process sequentially from 0 to N-2 (since loop is i < LEN_1D-1)
    for i in range(N - 1):
        # Load values for current iteration
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        e_val = tl.load(e_ptr + i)
        a_val = tl.load(a_ptr + i)
        
        # Compute operations
        # a[i+1] = b[i] + c[i]
        a_new = b_val + c_val
        tl.store(a_ptr + i + 1, a_new)
        
        # b[i] = c[i] * e[i]  
        b_new = c_val * e_val
        tl.store(b_ptr + i, b_new)
        
        # d[i] = a[i] * e[i]
        d_new = a_val * e_val
        tl.store(d_ptr + i, d_new)

def s3251_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Launch with single program since we need sequential processing
    grid = (1,)
    
    s3251_kernel[grid](
        a, b, c, d, e, N
    )
    
    return a, b, c, d, e