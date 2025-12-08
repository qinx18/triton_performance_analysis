import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must process sequentially due to j dependency
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    # Process all elements sequentially
    j = -1
    
    # Process elements one by one to maintain dependency
    for i in range(n_elements):
        # Load a[i]
        a_val = tl.load(a_ptr + i)
        
        # Check if a[i] > 0
        if a_val > 0.0:
            j += 1
            # Load b[j] and store to a[i]
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + i, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch kernel with single block to maintain sequential processing
    grid = (1,)
    s342_kernel[grid](a, b, n_elements, BLOCK_SIZE)