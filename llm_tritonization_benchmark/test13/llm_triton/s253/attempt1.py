import torch
import triton
import triton.language as tl

@triton.jit
def s253_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    condition = a_vals > b_vals
    s_vals = a_vals - b_vals * d_vals
    
    new_c_vals = tl.where(condition, c_vals + s_vals, c_vals)
    new_a_vals = tl.where(condition, s_vals, a_vals)
    
    tl.store(c_ptr + idx, new_c_vals, mask=mask)
    tl.store(a_ptr + idx, new_a_vals, mask=mask)

def s253_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s253_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)