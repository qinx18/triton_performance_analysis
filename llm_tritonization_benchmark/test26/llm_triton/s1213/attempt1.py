import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n):
    # This kernel has a strict loop-carried dependency
    # b[i] = a[i+1] * d[i] requires sequential processing
    # Only use one thread to process all elements sequentially
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process sequentially from i=1 to n-2
    for i in range(1, n - 1):
        # a[i] = b[i-1] + c[i]
        b_prev = tl.load(b_ptr + (i - 1))
        c_val = tl.load(c_ptr + i)
        a_val = b_prev + c_val
        tl.store(a_ptr + i, a_val)
        
        # b[i] = a[i+1] * d[i]
        a_next = tl.load(a_copy_ptr + (i + 1))
        d_val = tl.load(d_ptr + i)
        b_val = a_next * d_val
        tl.store(b_ptr + i, b_val)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    
    # Create read-only copy for WAR safety
    a_copy = a.clone()
    
    # Use single thread grid since this must be sequential
    grid = (1,)
    
    s1213_kernel[grid](
        a, a_copy, b, c, d, n
    )