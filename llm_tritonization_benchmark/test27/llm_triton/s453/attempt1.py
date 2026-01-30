import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    s = 0.0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        s += 2.0
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        result = s * b_vals
        
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s453_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    grid = (1,)
    
    s453_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a