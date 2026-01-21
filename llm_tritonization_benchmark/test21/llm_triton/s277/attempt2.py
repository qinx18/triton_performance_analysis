import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must process sequentially due to b[i+1] = ... dependency
    # Use single thread to ensure sequential execution
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Process all elements sequentially in a single thread
    for i in range(n - 1):
        # Load current values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Check if a[i] >= 0
        skip_all = a_val >= 0.0
        
        # Check if b[i] >= 0
        skip_a_update = b_val >= 0.0
        
        # Update a[i] if conditions allow
        should_update_a = (skip_all == 0) & (skip_a_update == 0)
        if should_update_a:
            a_new = a_val + c_val * d_val
            tl.store(a_ptr + i, a_new)
        
        # Update b[i+1] if not skipping all
        should_update_b = skip_all == 0
        if should_update_b:
            b_new = c_val + d_val * e_val
            tl.store(b_ptr + i + 1, b_new)

def s277_triton(a, b, c, d, e):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single thread due to sequential dependency
    grid = (1,)
    s277_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return a, b