import triton
import triton.language as tl
import torch

@triton.jit
def va_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(a_ptr + offsets, b, mask=mask)

def va_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    va_kernel[grid](a, b, n_elements, BLOCK_SIZE)
    
    return a