import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must process sequentially due to RAW dependency
    # Use single thread to process all elements
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Process sequentially from i=0 to n-2 (since loop goes to n-1)
    for i in range(n - 1):
        # Load scalars for current iteration
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        e_i = tl.load(e_ptr + i)
        a_i = tl.load(a_ptr + i)
        
        # Compute values
        a_next = b_i + c_i
        b_new = c_i * e_i
        d_new = a_i * e_i
        
        # Store results
        tl.store(a_ptr + i + 1, a_next)
        tl.store(b_ptr + i, b_new)
        tl.store(d_ptr + i, d_new)

def s3251_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Launch with single thread due to sequential dependency
    grid = (1,)
    BLOCK_SIZE = 1
    
    s3251_kernel[grid](
        a, b, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )