import torch
import triton
import triton.language as tl

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Sequential computation - process one element at a time
    for i in range(1, n_elements, 2):
        # Load a[i-1] and b[i]
        a_prev = tl.load(a_ptr + (i - 1))
        b_val = tl.load(b_ptr + i)
        
        # Compute and store a[i] = a[i-1] + b[i]
        result = a_prev + b_val
        tl.store(a_ptr + i, result)

def s111_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    
    # Launch kernel with single block since computation is sequential
    grid = (1,)
    s111_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )