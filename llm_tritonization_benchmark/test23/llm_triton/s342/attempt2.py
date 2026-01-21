import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the entire array sequentially since j depends on previous iterations
    # We'll use a single thread block to maintain the sequential dependency
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Initialize j counter
    j = -1
    
    # Process elements sequentially one by one
    for i in range(n):
        # Load a value
        a_val = tl.load(a_ptr + i)
        
        # Check if a[i] > 0
        if a_val > 0.0:
            j += 1
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + i, b_val)

def s342_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Since this has sequential dependencies, we use only one block
    grid = (1,)
    
    s342_kernel[grid](
        a, b, n,
        BLOCK_SIZE=BLOCK_SIZE
    )