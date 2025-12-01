import triton
import triton.language as tl
import torch

@triton.jit
def vpvts_kernel(a_ptr, b_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    result = a + b * s
    
    tl.store(a_ptr + offsets, result, mask=mask)

def vpvts_triton(a, b, s):
    n_elements = a.numel()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    vpvts_kernel[grid](a, b, s, n_elements, BLOCK_SIZE=BLOCK_SIZE)