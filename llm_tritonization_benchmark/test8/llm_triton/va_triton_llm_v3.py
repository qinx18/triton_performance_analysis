import torch
import triton
import triton.language as tl

@triton.jit
def va_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    tl.store(a_ptr + block_start + offsets, b_vals, mask=mask)

def va_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    va_kernel[grid](a, b, n_elements, BLOCK_SIZE)