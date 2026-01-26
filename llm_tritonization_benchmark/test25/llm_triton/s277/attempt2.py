import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel must be strictly sequential due to b[i+1] = ... b[i] dependency
    # Use single thread processing
    thread_id = tl.program_id(0)
    
    if thread_id != 0:
        return
    
    # Process all elements sequentially in a single thread
    for i in range(n_elements - 1):
        # Load current values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Check if a[i] >= 0
        skip_all = a_val >= 0.0
        
        # Check if b[i] >= 0 (only matters if not skipping all)
        skip_a_update = (b_val >= 0.0) & (~skip_all)
        
        # Update a[i] if both conditions are false
        update_a = (~skip_all) & (~skip_a_update)
        if update_a:
            a_new = a_val + c_val * d_val
            tl.store(a_ptr + i, a_new)
        
        # Update b[i+1] if not skipping all
        if ~skip_all:
            b_next_val = c_val + d_val * e_val
            tl.store(b_ptr + i + 1, b_next_val)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 1024
    grid = (1,)  # Single thread execution due to sequential dependency
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE,
    )
    
    return a, b