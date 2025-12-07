import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # This kernel must run sequentially due to RAW dependency
    # Use single thread to process all elements
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Process elements sequentially
    for i in range(n_elements - 1):
        # Load b[i], c[i], e[i], a[i]
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        e_i = tl.load(e_ptr + i)
        a_i = tl.load(a_ptr + i)
        
        # Compute operations
        a_next = b_i + c_i  # a[i+1] = b[i] + c[i]
        b_new = c_i * e_i   # b[i] = c[i] * e[i]
        d_new = a_i * e_i   # d[i] = a[i] * e[i]
        
        # Store results
        tl.store(a_ptr + i + 1, a_next)
        tl.store(b_ptr + i, b_new)
        tl.store(d_ptr + i, d_new)

def s3251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Launch with single thread due to sequential dependency
    grid = (1,)
    s3251_kernel[grid](
        a, b, c, d, e, n_elements
    )