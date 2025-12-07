import triton
import triton.language as tl
import torch

@triton.jit
def s253_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Conditional computation: if a[i] > b[i]
    cond = a_vals > b_vals
    
    # s = a[i] - b[i] * d[i]
    s = a_vals - b_vals * d_vals
    
    # Apply updates only where condition is true
    new_c = tl.where(cond, c_vals + s, c_vals)
    new_a = tl.where(cond, s, a_vals)
    
    tl.store(c_ptr + idx, new_c, mask=mask)
    tl.store(a_ptr + idx, new_a, mask=mask)

def s253_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s253_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )