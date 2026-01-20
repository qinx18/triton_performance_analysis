import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n):
    # This kernel must be strictly sequential due to RAW dependency
    # Use single thread to process all elements sequentially
    pid = tl.program_id(0)
    
    # Only process if this is thread 0
    if pid != 0:
        return
    
    # Process all iterations sequentially
    for i in range(n - 1):
        # Load values for current iteration
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        e_val = tl.load(e_ptr + i)
        a_val = tl.load(a_ptr + i)
        
        # Compute new values
        new_a = b_val + c_val
        new_b = c_val * e_val
        new_d = a_val * e_val
        
        # Store results
        tl.store(a_ptr + i + 1, new_a)
        tl.store(b_ptr + i, new_b)
        tl.store(d_ptr + i, new_d)

def s3251_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Launch single thread grid since computation must be sequential
    grid = (1,)
    
    s3251_kernel[grid](
        a, b, c, d, e, n
    )
    
    return a, b, c, d, e