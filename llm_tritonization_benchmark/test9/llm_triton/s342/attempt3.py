import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must run sequentially due to the j counter dependency
    block_id = tl.program_id(0)
    
    # Only use the first block
    if block_id > 0:
        return
    
    j = -1
    
    # Process elements sequentially
    for i in range(n_elements):
        a_val = tl.load(a_ptr + i)
        
        if a_val > 0.0:
            j += 1
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + i, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch with single block due to sequential dependency
    grid = (1,)
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a