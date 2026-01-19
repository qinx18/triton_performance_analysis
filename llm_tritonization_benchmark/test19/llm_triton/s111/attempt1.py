import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel must run sequentially due to loop-carried dependency
    # Use single thread to process all elements
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process sequentially: a[i] = a[i-1] + b[i] for i in range(1, N, 2)
    for i in range(1, N, 2):
        # Load a[i-1] and b[i]
        a_prev = tl.load(a_ptr + (i - 1))
        b_curr = tl.load(b_ptr + i)
        
        # Compute and store a[i]
        result = a_prev + b_curr
        tl.store(a_ptr + i, result)

def s111_triton(a, b):
    N = a.shape[0]
    
    # Use single thread since computation must be sequential
    grid = (1,)
    BLOCK_SIZE = 256
    
    s111_kernel[grid](
        a, b, N, 
        BLOCK_SIZE=BLOCK_SIZE
    )