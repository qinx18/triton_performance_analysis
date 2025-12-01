import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Sequential computation due to dependency
    for i in range(n_elements):
        if i == 0:
            im1_val = tl.load(b_ptr + n_elements - 1)  # b[LEN_1D-1]
        else:
            im1_val = tl.load(b_ptr + i - 1)  # b[i-1]
        
        b_val = tl.load(b_ptr + i)
        result = (b_val + im1_val) * 0.5
        tl.store(a_ptr + i, result)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (1,)  # Single block due to sequential dependency
    
    s291_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )