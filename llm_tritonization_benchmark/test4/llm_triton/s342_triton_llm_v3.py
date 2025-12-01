import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel needs to be sequential due to data dependencies
    # We'll use a single thread to process the entire array
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    j = -1
    
    # Process elements sequentially
    for i in range(n):
        a_val = tl.load(a_ptr + i)
        if a_val > 0.0:
            j += 1
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + i, b_val)

def s342_triton(a, b):
    n = a.shape[0]
    
    # Launch with single thread since this is inherently sequential
    grid = (1,)
    BLOCK_SIZE = 1024
    
    s342_kernel[grid](
        a, b, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a