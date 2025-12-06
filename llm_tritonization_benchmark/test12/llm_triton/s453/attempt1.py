import torch
import triton
import triton.language as tl

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential computation - each element depends on previous
    for i in range(n_elements):
        if i < n_elements:
            s = (i + 1) * 2.0  # s accumulates 2.0 for each iteration
            b_val = tl.load(b_ptr + i)
            result = s * b_val
            tl.store(a_ptr + i, result)

def s453_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch single block since computation is sequential
    grid = (1,)
    
    s453_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a