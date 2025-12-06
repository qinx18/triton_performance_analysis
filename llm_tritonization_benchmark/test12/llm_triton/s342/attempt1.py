import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Sequential processing with one thread block
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially to maintain dependencies
    j = -1
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
    BLOCK_SIZE = 128
    
    # Launch with single block since this is inherently sequential
    grid = (1,)
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a