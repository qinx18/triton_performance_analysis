import triton
import triton.language as tl
import torch

@triton.jit
def s471_kernel(b_ptr, c_ptr, d_ptr, e_ptr, x_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < m
    
    # Load arrays
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # x[i] = b[i] + d[i] * d[i]
    x_vals = b_vals + d_vals * d_vals
    tl.store(x_ptr + indices, x_vals, mask=mask)
    
    # b[i] = c[i] + d[i] * e[i]
    b_new = c_vals + d_vals * e_vals
    tl.store(b_ptr + indices, b_new, mask=mask)

def s471_triton(b, c, d, e, x, m):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s471_kernel[grid](b, c, d, e, x, m, BLOCK_SIZE)
    
    return b, x