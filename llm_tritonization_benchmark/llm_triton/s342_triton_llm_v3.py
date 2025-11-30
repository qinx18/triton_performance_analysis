import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel cannot be parallelized due to data dependencies
    # j value depends on previous iterations, so we must process sequentially
    
    # Only process from thread 0 to maintain sequential order
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    j = -1
    
    # Process all elements sequentially
    for i in range(n_elements):
        a_val = tl.load(a_ptr + i)
        if a_val > 0.0:
            j += 1
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + i, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    
    # Use single thread since this cannot be parallelized
    grid = (1,)
    BLOCK_SIZE = 1024
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a