import triton
import triton.language as tl
import torch

@triton.jit
def s271_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    condition = b_vals > 0.0
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    new_vals = a_vals + b_vals * c_vals
    
    final_vals = tl.where(condition, new_vals, a_vals)
    
    tl.store(a_ptr + offsets, final_vals, mask=mask)

def s271_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s271_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return a