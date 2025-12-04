import torch
import triton
import triton.language as tl

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    a_offsets = block_start + offsets
    j_offsets = a_offsets + 1
    
    mask_a = a_offsets < n_elements
    mask_j = j_offsets < (n_elements + 1)
    
    a_j_vals = tl.load(a_ptr + j_offsets, mask=mask_j, other=0.0)
    b_vals = tl.load(b_ptr + a_offsets, mask=mask_a, other=0.0)
    
    result = a_j_vals + b_vals
    
    tl.store(a_ptr + a_offsets, result, mask=mask_a)

def s121_triton(a, b):
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s121_kernel[grid](a, b, n_elements, BLOCK_SIZE)