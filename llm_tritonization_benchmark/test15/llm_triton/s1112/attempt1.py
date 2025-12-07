import triton
import triton.language as tl
import torch

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    result = b_vals + 1.0
    tl.store(a_ptr + indices, result, mask=mask)

def s1112_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1112_kernel[grid](a, b, n_elements, BLOCK_SIZE)