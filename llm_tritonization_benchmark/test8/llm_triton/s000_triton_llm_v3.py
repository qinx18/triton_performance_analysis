import torch
import triton
import triton.language as tl

@triton.jit
def s000_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    result = b_vals + 1.0
    
    tl.store(a_ptr + indices, result, mask=mask)

def s000_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s000_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )