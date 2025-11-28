import triton
import triton.language as tl
import torch

@triton.jit
def s171_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    a_offsets = offsets * inc
    a_vals = tl.load(a_ptr + a_offsets, mask=mask)
    
    result = a_vals + b_vals
    tl.store(a_ptr + a_offsets, result, mask=mask)

def s171_triton(a, b, inc):
    n_elements = a.shape[0] // inc
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s171_kernel[grid](a, b, inc, n_elements, BLOCK_SIZE)