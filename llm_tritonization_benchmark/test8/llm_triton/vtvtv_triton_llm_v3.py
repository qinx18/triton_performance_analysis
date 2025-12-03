import torch
import triton
import triton.language as tl

@triton.jit
def vtvtv_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    
    result = a_vals * b_vals * c_vals
    
    tl.store(a_ptr + current_offsets, result, mask=mask)

def vtvtv_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vtvtv_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)