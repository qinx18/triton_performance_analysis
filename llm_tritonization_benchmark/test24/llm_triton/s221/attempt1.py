import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n):
    # This kernel must run sequentially due to b[i] = b[i-1] + ... dependency
    # Use single thread to process all elements
    if tl.program_id(0) != 0:
        return
    
    # Process sequentially from i=1 to n-1
    for i in range(1, n):
        # Load c[i] and d[i]
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Load current a[i]
        a_val = tl.load(a_ptr + i)
        
        # Update a[i] += c[i] * d[i]
        a_val = a_val + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Load b[i-1] for sequential dependency
        b_prev = tl.load(b_ptr + i - 1)
        
        # Compute b[i] = b[i-1] + a[i] + d[i]
        b_val = b_prev + a_val + d_val
        tl.store(b_ptr + i, b_val)

def s221_triton(a, b, c, d):
    n = a.shape[0]
    
    # Use single thread since computation must be sequential
    grid = (1,)
    
    s221_kernel[grid](
        a, b, c, d, n
    )
    
    return a, b