import torch
import triton
import triton.language as tl

@triton.jit
def s351_kernel(a_ptr, b_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    element_offsets = block_start + offsets
    mask = element_offsets < n_elements
    
    a_vals = tl.load(a_ptr + element_offsets, mask=mask)
    b_vals = tl.load(b_ptr + element_offsets, mask=mask)
    
    result = a_vals + alpha * b_vals
    tl.store(a_ptr + element_offsets, result, mask=mask)

def s351_triton(a, b, c):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s351_kernel[grid](
        a, b, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )