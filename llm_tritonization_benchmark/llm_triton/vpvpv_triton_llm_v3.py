import triton
import triton.language as tl
import torch

@triton.jit
def vpvpv_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    a = tl.load(a_ptr + offsets, mask=mask)
    
    result = a + b + c
    tl.store(a_ptr + offsets, result, mask=mask)

def vpvpv_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vpvpv_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)