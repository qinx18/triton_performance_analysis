import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements):
    # This kernel must run sequentially due to RAW dependency
    # Use single thread to process all elements
    tid = tl.program_id(0)
    
    if tid > 0:
        return
    
    # Process elements sequentially from index 1 to n_elements-1
    for i in range(1, n_elements):
        # Load c[i] and d[i]
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Load a[i]
        a_val = tl.load(a_ptr + i)
        
        # Update a[i] += c[i] * d[i]
        a_val = a_val + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Load b[i-1] and b[i]
        b_prev = tl.load(b_ptr + i - 1)
        
        # Update b[i] = b[i-1] + a[i] + d[i]
        b_val = b_prev + a_val + d_val
        tl.store(b_ptr + i, b_val)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    
    s221_kernel[grid](
        a, b, c, d,
        n_elements
    )
    
    return a, b