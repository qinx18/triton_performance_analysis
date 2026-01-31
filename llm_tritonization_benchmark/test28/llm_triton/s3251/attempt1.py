import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # This kernel must process sequentially due to RAW dependency
    # Use single thread approach
    pid = tl.program_id(0)
    
    # Only first thread does the work to ensure sequential processing
    if pid == 0:
        for i in range(n_elements - 1):
            # Load values for current iteration
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            e_val = tl.load(e_ptr + i)
            a_val = tl.load(a_ptr + i)
            
            # Compute and store results
            a_new = b_val + c_val
            b_new = c_val * e_val  
            d_new = a_val * e_val
            
            tl.store(a_ptr + i + 1, a_new)
            tl.store(b_ptr + i, b_new)
            tl.store(d_ptr + i, d_new)

def s3251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Launch with single thread due to sequential dependency
    grid = (1,)
    
    s3251_kernel[grid](
        a, b, c, d, e,
        n_elements
    )
    
    return a, b, c, d, e