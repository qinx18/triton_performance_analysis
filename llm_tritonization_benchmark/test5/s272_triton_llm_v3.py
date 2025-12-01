import torch
import triton
import triton.language as tl

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    condition = e_vals >= t
    
    cd_product = c_vals * d_vals
    cc_product = c_vals * c_vals
    
    new_a = tl.where(condition, a_vals + cd_product, a_vals)
    new_b = tl.where(condition, b_vals + cc_product, b_vals)
    
    tl.store(a_ptr + offsets, new_a, mask=mask)
    tl.store(b_ptr + offsets, new_b, mask=mask)

def s272_triton(a, b, c, d, e, t):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s272_kernel[grid](
        a, b, c, d, e,
        t, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )