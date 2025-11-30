import triton
import triton.language as tl
import torch

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Sequential computation due to dependency
    for i in range(n_elements):
        if i == 0:
            im1 = n_elements - 1
        else:
            im1 = i - 1
        
        b_i = tl.load(b_ptr + i)
        b_im1 = tl.load(b_ptr + im1)
        result = (b_i + b_im1) * 0.5
        tl.store(a_ptr + i, result)

def s291_triton(a, b):
    n_elements = a.shape[0]
    
    # Use single thread due to sequential dependency
    BLOCK_SIZE = n_elements
    grid = (1,)
    
    s291_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )