import torch
import triton
import triton.language as tl

@triton.jit
def s000_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    result = b_vals + 1.0
    
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s000_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s000_kernel[grid](
        a, b, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )