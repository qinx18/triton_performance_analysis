import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Sequential processing due to dependency on j
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize j as a scalar
    j = -1
    
    # Process elements sequentially
    for i in range(n_elements):
        # Load single elements as scalars
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Increment j
        j += 1
        
        # Compute result based on condition
        if b_val > 0.0:
            result = b_val + d_val * e_val
        else:
            result = c_val + d_val * e_val
        
        # Store at position j
        tl.store(a_ptr + j, result)

def s124_triton(a, b, c, d, e):
    n_elements = b.numel()
    
    # Use single thread for sequential processing
    grid = (1,)
    BLOCK_SIZE = 1
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a