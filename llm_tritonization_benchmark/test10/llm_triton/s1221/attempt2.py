import triton
import triton.language as tl
import torch

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Process sequentially due to dependency
    for i in range(4, n_elements):
        b_prev = tl.load(b_ptr + i - 4)
        a_val = tl.load(a_ptr + i)
        result = b_prev + a_val
        tl.store(b_ptr + i, result)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single thread
    s1221_kernel[(1,)](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return b