import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must run sequentially due to the loop-carried dependency
    # a[i+1] = b[i] + c[i] where a[i+1] depends on previous iterations
    
    # Use single thread to process sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process all elements sequentially in a single thread
    for i in range(n - 1):
        # Load b[i], c[i], e[i], a[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        e_val = tl.load(e_ptr + i)
        a_val = tl.load(a_ptr + i)
        
        # Compute and store results
        # a[i+1] = b[i] + c[i]
        tl.store(a_ptr + (i + 1), b_val + c_val)
        
        # b[i] = c[i] * e[i]
        tl.store(b_ptr + i, c_val * e_val)
        
        # d[i] = a[i] * e[i]
        tl.store(d_ptr + i, a_val * e_val)

def s3251_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Use single thread grid since we must process sequentially
    grid = (1,)
    BLOCK_SIZE = 256
    
    s3251_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return a, b, c, d, e