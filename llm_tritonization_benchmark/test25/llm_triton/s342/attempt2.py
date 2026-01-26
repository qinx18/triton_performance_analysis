import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially
    pid = tl.program_id(0)
    
    j = -1
    
    # Process elements sequentially
    for i in range(n_elements):
        a_val = tl.load(a_ptr + i)
        
        if a_val > 0.0:
            j = j + 1
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + i, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1
    
    # Launch kernel with single thread since we need sequential processing
    grid = (1,)
    s342_kernel[grid](a, b, n_elements, BLOCK_SIZE)
    
    return a