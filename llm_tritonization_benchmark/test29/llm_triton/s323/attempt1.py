import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, N):
    # This kernel must process sequentially due to loop-carried dependency
    # Use only one thread to ensure correct sequential execution
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Process sequentially from i=1 to N-1
    for i in range(1, N):
        # Load b[i-1]
        b_prev = tl.load(b_ptr + (i - 1))
        
        # Load c[i], d[i], e[i]
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_val = b_prev + c_val * d_val
        
        # Store a[i]
        tl.store(a_ptr + i, a_val)
        
        # Compute b[i] = a[i] + c[i] * e[i]
        b_val = a_val + c_val * e_val
        
        # Store b[i]
        tl.store(b_ptr + i, b_val)

def s323_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Launch with only 1 thread since computation must be sequential
    grid = (1,)
    
    s323_kernel[grid](
        a, b, c, d, e, N
    )
    
    return a, b