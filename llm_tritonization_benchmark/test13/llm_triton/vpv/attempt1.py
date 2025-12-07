import triton
import triton.language as tl
import torch

@triton.jit
def vpv_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    
    result = a_vals + b_vals
    
    tl.store(a_ptr + current_offsets, result, mask=mask)

def vpv_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vpv_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )