import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must run with a single thread since the computation is strictly sequential
    # due to the a[i+1] = ... dependency on a[i] from previous iteration
    
    pid = tl.program_id(0)
    if pid != 0:  # Only run on the first program
        return
    
    # Process all elements sequentially
    for i in range(n - 1):
        # Load values for current iteration
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        e_val = tl.load(e_ptr + i)
        a_val = tl.load(a_ptr + i)
        
        # Compute updates
        a_new = b_val + c_val
        b_new = c_val * e_val
        d_new = a_val * e_val
        
        # Store results
        tl.store(a_ptr + i + 1, a_new)
        tl.store(b_ptr + i, b_new)
        tl.store(d_ptr + i, d_new)

def s3251_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Launch with single program since computation is strictly sequential
    grid = (1,)
    
    s3251_kernel[grid](
        a, b, c, d, e,
        n,
        BLOCK_SIZE=1
    )