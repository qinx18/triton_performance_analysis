import torch
import triton
import triton.language as tl

@triton.jit
def s2712_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    condition = a_vals > b_vals
    update_vals = b_vals * c_vals
    new_a_vals = tl.where(condition, a_vals + update_vals, a_vals)
    
    tl.store(a_ptr + indices, new_a_vals, mask=mask)

def s2712_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2712_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)