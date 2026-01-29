import triton
import triton.language as tl
import torch

@triton.jit
def s471_kernel(
    b_ptr, c_ptr, d_ptr, e_ptr, x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load input values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    
    # Compute x[i] = b[i] + d[i] * d[i]
    x_vals = b_vals + d_vals * d_vals
    tl.store(x_ptr + current_offsets, x_vals, mask=mask)
    
    # Compute b[i] = c[i] + d[i] * e[i]
    b_new = c_vals + d_vals * e_vals
    tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s471_triton(b, c, d, e, x, m):
    n_elements = m
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s471_kernel[grid](
        b, c, d, e, x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )