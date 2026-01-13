import torch
import triton
import triton.language as tl

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    e_vals = tl.load(e_ptr + indices, mask=mask)
    condition = e_vals >= t
    
    valid_mask = mask & condition
    
    a_vals = tl.load(a_ptr + indices, mask=valid_mask)
    c_vals = tl.load(c_ptr + indices, mask=valid_mask)
    d_vals = tl.load(d_ptr + indices, mask=valid_mask)
    b_vals = tl.load(b_ptr + indices, mask=valid_mask)
    
    a_new = a_vals + c_vals * d_vals
    b_new = b_vals + c_vals * c_vals
    
    tl.store(a_ptr + indices, a_new, mask=valid_mask)
    tl.store(b_ptr + indices, b_new, mask=valid_mask)

def s272_triton(a, b, c, d, e, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s272_kernel[grid](
        a, b, c, d, e, 
        t, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )