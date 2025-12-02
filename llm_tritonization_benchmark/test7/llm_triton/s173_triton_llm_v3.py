import torch
import triton
import triton.language as tl

@triton.jit
def s173_kernel(a_ptr, b_ptr, k, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    result = a_vals + b_vals
    
    output_indices = indices + k
    tl.store(a_ptr + output_indices, result, mask=mask)

def s173_triton(a, b, k):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s173_kernel[grid](
        a, b, k, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )