import torch
import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Sequential processing required due to dependencies
    for i in range(n_elements):
        if i == 0:
            im1 = n_elements - 1
            im2 = n_elements - 2
        elif i == 1:
            im1 = 0
            im2 = n_elements - 1
        else:
            im1 = i - 1
            im2 = i - 2
            
        b_i = tl.load(b_ptr + i)
        b_im1 = tl.load(b_ptr + im1)
        b_im2 = tl.load(b_ptr + im2)
        
        result = (b_i + b_im1 + b_im2) * 0.333
        tl.store(a_ptr + i, result)

def s292_triton(a, b):
    n_elements = a.shape[0]
    
    # Use single thread due to sequential dependencies
    BLOCK_SIZE = 1
    grid = (1,)
    
    s292_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )