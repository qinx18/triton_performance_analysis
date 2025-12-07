import torch
import triton
import triton.language as tl

@triton.jit
def s271_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    condition = b_vals > 0.0
    combined_mask = mask & condition
    
    a_vals = tl.load(a_ptr + indices, mask=combined_mask)
    c_vals = tl.load(c_ptr + indices, mask=combined_mask)
    
    result = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + indices, result, mask=combined_mask)

def s271_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s271_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)