import torch
import triton
import triton.language as tl

@triton.jit
def s251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    s_vals = b_vals + c_vals * d_vals
    a_vals = s_vals * s_vals
    
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s251_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s251_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a