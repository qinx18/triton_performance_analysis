import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Sequential processing - single thread handles all iterations
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process sequentially from i=1 to n-2
    for i in range(1, n - 1):
        # Load values
        b_prev = tl.load(b_ptr + (i - 1))
        c_val = tl.load(c_ptr + i)
        a_next = tl.load(a_copy_ptr + (i + 1))
        d_val = tl.load(d_ptr + i)
        
        # Compute
        a_new = b_prev + c_val
        b_new = a_next * d_val
        
        # Store results
        tl.store(a_ptr + i, a_new)
        tl.store(b_ptr + i, b_new)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    
    # Create read-only copy for WAR safety
    a_copy = a.clone()
    
    # Use single thread for sequential processing
    grid = (1,)
    BLOCK_SIZE = 256
    
    s1213_kernel[grid](a, a_copy, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)