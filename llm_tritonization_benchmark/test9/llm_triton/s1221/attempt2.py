import torch
import triton
import triton.language as tl

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    current_offsets = block_start + offsets + 4
    mask = current_offsets < n_elements
    
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    b_prev_vals = tl.load(b_ptr + current_offsets - 4, mask=mask)
    
    result = b_prev_vals + a_vals
    
    tl.store(b_ptr + current_offsets, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    n_work = n_elements - 4
    if n_work <= 0:
        return b
    
    grid = (triton.cdiv(n_work, BLOCK_SIZE),)
    
    s1221_kernel[grid](a, b, n_elements, BLOCK_SIZE)
    
    return b