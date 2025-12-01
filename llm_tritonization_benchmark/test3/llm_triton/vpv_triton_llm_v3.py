import torch
import triton
import triton.language as tl

@triton.jit
def vpv_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    result = a + b
    
    tl.store(a_ptr + offsets, result, mask=mask)

def vpv_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vpv_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)