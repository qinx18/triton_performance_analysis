import torch
import triton
import triton.language as tl

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # Sequential processing - must use single thread due to RAW dependency
    pid = tl.program_id(0)
    
    # Only first thread processes to maintain sequential dependency
    if pid != 0:
        return
    
    # Process sequentially from i=0 to i=n_elements-2
    for i in range(n_elements - 1):
        # Load current values
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        e_i = tl.load(e_ptr + i)
        a_i = tl.load(a_ptr + i)
        
        # Compute updates
        a_new = b_i + c_i
        b_new = c_i * e_i
        d_new = a_i * e_i
        
        # Store results
        tl.store(a_ptr + i + 1, a_new)
        tl.store(b_ptr + i, b_new)
        tl.store(d_ptr + i, d_new)

def s3251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Launch with single thread block to maintain sequential execution
    grid = (1,)
    s3251_kernel[grid](a, b, c, d, e, n_elements)
    
    return a, b, c, d, e