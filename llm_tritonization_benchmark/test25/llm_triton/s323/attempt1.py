import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n):
    # This kernel must run sequentially due to loop-carried dependency
    # Use single thread to process all elements
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Sequential loop processing
    for i in range(1, n):
        # Load b[i-1]
        b_prev = tl.load(b_ptr + (i - 1))
        
        # Load c[i], d[i], e[i]
        c_i = tl.load(c_ptr + i)
        d_i = tl.load(d_ptr + i)
        e_i = tl.load(e_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_i = b_prev + c_i * d_i
        
        # Store a[i]
        tl.store(a_ptr + i, a_i)
        
        # Compute b[i] = a[i] + c[i] * e[i]
        b_i = a_i + c_i * e_i
        
        # Store b[i]
        tl.store(b_ptr + i, b_i)

def s323_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Launch single thread grid
    grid = (1,)
    
    s323_kernel[grid](
        a, b, c, d, e, n
    )