import triton
import triton.language as tl
import torch

@triton.jit
def s471_kernel(
    x_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    b_val = tl.load(b_ptr + offsets, mask=mask)
    d_val = tl.load(d_ptr + offsets, mask=mask)
    c_val = tl.load(c_ptr + offsets, mask=mask)
    e_val = tl.load(e_ptr + offsets, mask=mask)
    
    # x[i] = b[i] + d[i] * d[i]
    x_val = b_val + d_val * d_val
    tl.store(x_ptr + offsets, x_val, mask=mask)
    
    # b[i] = c[i] + d[i] * e[i]
    b_new = c_val + d_val * e_val
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s471_triton(x, b, c, d, e):
    n_elements = x.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s471_kernel[grid](
        x, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x, b