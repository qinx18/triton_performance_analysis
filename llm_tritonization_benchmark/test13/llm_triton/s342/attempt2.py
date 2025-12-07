import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must process sequentially due to j dependency
    # Process one element at a time to maintain correct ordering
    
    j = -1
    
    for i in range(n_elements):
        a_val = tl.load(a_ptr + i)
        
        if a_val > 0.0:
            j = j + 1
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + i, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single program since we need sequential processing
    grid = (1,)
    
    s342_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a